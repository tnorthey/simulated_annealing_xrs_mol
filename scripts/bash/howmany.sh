#!/bin/bash
#
ls $1 | sed -n 's/^\([0-9][0-9]\)_.*/\1/p' | sort | uniq -c

