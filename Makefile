.PHONY: build train up down logs dash status flatten test
build:   ; docker compose build
train:   ; docker compose run --rm bot python -m app.train
up:      ; docker compose up -d && echo "dashboard: http://localhost:8080"
down:    ; docker compose stop && docker compose down
logs:    ; docker compose logs -f bot
status:  ; @cat state/heartbeat.json 2>/dev/null || echo "no heartbeat yet"
flatten: ; docker compose run --rm bot python -c "from app.settings import S; from app.broker import Broker; Broker(S).flatten_all('manual')"
test:    ; docker compose run --rm bot python -m app.selftest
