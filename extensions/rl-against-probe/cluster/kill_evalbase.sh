#!/bin/bash
pkill -u user -f "run_evalbase[.]sh"
pkill -u user -f "sample[.]py ib_eval_tasks[.]jsonl p0_eval_"
