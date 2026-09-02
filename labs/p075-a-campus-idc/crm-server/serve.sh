#!/bin/sh
while IFS= read -r line; do
  [ "$line" = "$(printf '\r')" ] && break
done
printf 'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 7\r\nConnection: close\r\n\r\nCRM_OK\n'

