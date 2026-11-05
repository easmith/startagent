#!/bin/sh
set -eu

src_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
dest_dir=${1:-${STARTAGENT_INSTALL_DIR:-"${HOME}/.local/bin"}}

mkdir -p "$dest_dir"
cp "$src_dir/startagent.py" "$dest_dir/startagent"
chmod 755 "$dest_dir/startagent"

printf 'installed %s\n' "$dest_dir/startagent"
