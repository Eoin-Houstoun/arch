"""Content manifest of a directory tree (stdlib only), failing closed.

    python manifest.py <dir> [--exclude <relpath>] [--base]

--base also hashes the Python installation the running interpreter links to
(sys.base_prefix), so a venv's symlinked interpreter and stdlib are covered.

Hashes every entry, not just regular files: its relative path, type, mode,
the target of a symlink, and the bytes of a regular file. Any entry that
cannot be read is an error (exit 1), never skipped.
"""
import hashlib
import os
import stat
import sys


def manifest(root, exclude=()):
    h = hashlib.sha256()
    for d, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(dirs + files):
            p = os.path.join(d, name)
            rel = os.path.relpath(p, root)
            if rel in exclude:
                continue
            st = os.lstat(p)
            h.update(("%s\0%o\0" % (rel, st.st_mode)).encode())
            if stat.S_ISLNK(st.st_mode):
                h.update(os.readlink(p).encode() + b"\0")
            elif stat.S_ISREG(st.st_mode):
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
            elif not stat.S_ISDIR(st.st_mode):
                raise OSError("unexpected entry type at %s" % rel)
    return h.hexdigest()


if __name__ == "__main__":
    args = sys.argv[1:]
    excl = tuple(args[i + 1] for i, a in enumerate(args) if a == "--exclude")
    try:
        out = manifest(args[0], excl)
        if "--base" in args:
            out = hashlib.sha256((out + manifest(os.path.realpath(sys.base_prefix))).encode()).hexdigest()
        print(out)
    except (OSError, IndexError) as exc:
        sys.exit("manifest: %s" % exc)
