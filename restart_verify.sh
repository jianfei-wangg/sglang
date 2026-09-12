#!/usr/bin/env bash
set -euo pipefail
cd /cpfs/user/qianwu/sglang-qianwu
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/python"

SUMMARY="${SUMMARY:-/cpfs/user/qianwu/sglang-qianwu/restart_verify_summary.log}"
: > "$SUMMARY"
NUM_RESTARTS="${NUM_RESTARTS:-5}"
RUNS_PER_RESTART="${RUNS_PER_RESTART:-2}"
MODEL_PATH="${MODEL_PATH:-/cpfs/user/qianwu/models/dots.note.vlm.iter5961_ata_fp8_infra}"
PYTHON="${PYTHON:-/root/.pyenv/versions/bh-ve/bin/python}"
PORT="${PORT:-30000}"

kill_server() {
  pgrep -af "sglang serve|launch_server" | awk '{print $1}' | xargs -r kill 2>/dev/null || true
  sleep 5
  for _ in $(seq 1 30); do
    if ! pgrep -f "sglang serve|launch_server" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  pgrep -af "sglang serve|launch_server" | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
  sleep 3
}

wait_ready() {
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then
      echo "ready ${i}0s" | tee -a "$SUMMARY"
      return 0
    fi
    sleep 10
  done
  return 1
}

start_server() {
  "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --context-length 393216 \
    --enable-dp-attention \
    --dp-size 8 \
    --tp-size 8 \
    --ep-size 8 \
    --mem-fraction-static 0.87 \
    --max-running-requests 256 \
    --chunked-prefill-size 16384 \
    --trust-remote-code \
    --swa-full-tokens-ratio 0.03 \
    --page-size 64 \
    --moe-dense-tp-size 1 \
    --watchdog-timeout 1800 \
    --cuda-graph-backend-decode full \
    --cuda-graph-backend-prefill disabled \
    --cuda-graph-max-bs-decode 32 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --moe-a2a-backend deepep \
    --moe-runner-backend auto \
    --deepep-dispatcher-output-dtype auto \
    --deepep-mode auto \
    --enable-nccl-nvls \
    --enable-multimodal \
    --enable-metrics \
    "$@" &
}

for cycle in $(seq 1 "$NUM_RESTARTS"); do
  LOG="restart_verify_cycle${cycle}.log"
  echo "======== CYCLE $cycle/$NUM_RESTARTS $(date -Iseconds) ========" | tee -a "$SUMMARY"
  kill_server
  echo "starting server cycle $cycle..." | tee -a "$SUMMARY"
  start_server > "$LOG" 2>&1
  if ! wait_ready; then
    echo "FAIL: server not ready cycle $cycle" | tee -a "$SUMMARY"
    tail -30 "$LOG" | tee -a "$SUMMARY"
    exit 1
  fi
  start_lines=$(wc -l < "$LOG")
  for run in $(seq 1 "$RUNS_PER_RESTART"); do
    echo "--- cycle $cycle run $run ---" | tee -a "$SUMMARY"
    if ! "$PYTHON" -m sglang.test.few_shot_gsm8k --port "$PORT" --num-questions 3000 \
        2>&1 | tee "/tmp/gsm8k_c${cycle}_r${run}.log" | tail -3 | tee -a "$SUMMARY"; then
      echo "FAIL: gsm8k client error c$cycle r$run" | tee -a "$SUMMARY"
      exit 1
    fi
    sleep 5
    if tail -n +"$start_lines" "$LOG" | grep -q "pool memory leak"; then
      echo "FAIL: LEAK cycle $cycle run $run" | tee -a "$SUMMARY"
      tail -n +"$start_lines" "$LOG" | grep -A1 "pool memory leak" | tee -a "$SUMMARY"
      exit 1
    fi
    if tail -n +"$start_lines" "$LOG" | grep -q "Scheduler hit an exception"; then
      echo "FAIL: scheduler exception cycle $cycle run $run" | tee -a "$SUMMARY"
      exit 1
    fi
    h=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health")
    if [ "$h" != "200" ]; then
      echo "FAIL: health=$h cycle $cycle run $run" | tee -a "$SUMMARY"
      exit 1
    fi
    echo "OK cycle $cycle run $run" | tee -a "$SUMMARY"
  done
  echo "CYCLE $cycle PASSED" | tee -a "$SUMMARY"
done

echo "ALL $NUM_RESTARTS RESTARTS x $RUNS_PER_RESTART RUNS PASSED" | tee -a "$SUMMARY"
