# Build as yourself, install with sudo:  make && sudo make install
# install only copies; it never builds, so Go is not run as root.
PREFIX  ?= /usr/local
ETC     := $(PREFIX)/etc/gachbothar
LOGDIR  := /var/log/gachbothar
# The service runs as the user who ran sudo, unless SVCUSER is given.
SVCUSER ?= $(or $(SUDO_USER),$(USER))
VERSION := $(shell git describe --always --dirty 2>/dev/null || echo dev)

.PHONY: build test install uninstall

build:
	CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=$(VERSION)" -o build/gachbothar ./src

test:
	go test -count=1 ./...

install:
	@test -x build/gachbothar || { echo "run 'make' first"; exit 1; }
	@if [ -f $(PREFIX)/bin/gachbothar ]; then cp -p $(PREFIX)/bin/gachbothar $(PREFIX)/bin/gachbothar.prev; fi
	install -D -m 0755 build/gachbothar $(PREFIX)/bin/gachbothar
	install -d -m 0755 $(ETC)
	@# The live config is never overwritten; edit it in place. A first install starts from the example.
	@if [ ! -f $(ETC)/providers.json ]; then install -m 0644 packaging/providers.json.example $(ETC)/providers.json; fi
	install -d -m 0755 -o $(SVCUSER) $(LOGDIR)
	install -m 0644 packaging/gachbothar.logrotate /etc/logrotate.d/gachbothar
	sed 's/@USER@/$(SVCUSER)/' packaging/gachbothar.service > /etc/systemd/system/gachbothar.service
	systemctl daemon-reload
	systemctl enable gachbothar
	systemctl restart gachbothar

uninstall:
	systemctl disable --now gachbothar || true
	rm -f /etc/systemd/system/gachbothar.service /etc/logrotate.d/gachbothar
	rm -f $(PREFIX)/bin/gachbothar $(PREFIX)/bin/gachbothar.prev
	systemctl daemon-reload
	@echo "kept $(ETC) and $(LOGDIR)"
