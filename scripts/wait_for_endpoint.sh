#!/bin/bash
# Poll until the 70B endpoint is STARTED, then exit 0.
# Tries to start it every 60s if it's STOPPED and capacity is unavailable.

set -euo pipefail
ENDPOINT_ID="${1:-endpoint-93c31471-7a69-43d3-b6c7-98586b9d1cf2}"

while true; do
  STATE=$(curl -s -H "Authorization: Bearer $TOGETHER_API_KEY" \
    "https://api.together.xyz/v1/endpoints/${ENDPOINT_ID}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('state','?'))")

  echo "$(date +%H:%M:%S) state=$STATE"

  case "$STATE" in
    STARTED)
      # Confirm replica ready
      READY=$(curl -s -H "Authorization: Bearer $TOGETHER_API_KEY" \
        "https://api.together.xyz/v1/endpoints/${ENDPOINT_ID}" \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('autoscaling',{}).get('ready_replicas',0))")
      if [ "$READY" = "1" ]; then
        echo "READY"
        exit 0
      fi
      ;;
    STOPPED)
      echo "  attempting to start..."
      RESP=$(curl -s -X PATCH \
        -H "Authorization: Bearer $TOGETHER_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"state":"STARTED"}' \
        "https://api.together.xyz/v1/endpoints/${ENDPOINT_ID}")
      echo "  $(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',{}).get('message','OK -- state='+d.get('state','?')))")"
      ;;
    STARTING|PENDING)
      ;;
    *)
      echo "  unexpected state, will retry"
      ;;
  esac
  sleep 60
done
