#!/bin/sh
while IFS= read -r line; do
  [ "$line" = "$(printf '\r')" ] && break
done
printf 'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 8\r\nConnection: close\r\n\r\nWIKI_OK\n'
