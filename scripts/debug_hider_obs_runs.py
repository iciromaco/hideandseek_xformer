#!/usr/bin/env python3
import runpy

for i in range(5):
    print('\n=== RUN', i+1, '===')
    runpy.run_path('scripts/debug_hider_obs.py', run_name='__main__')
