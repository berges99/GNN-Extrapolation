# The Simplicity Bias of Graph Neural Networks







### Teacher Setting

---

The teacher setting ...

```bash
python3 run_teacher.py --dataset_filename ) [DATASET_FILENAME]
                       --initial_relabeling ) {'ones', 'degrees'}
                       # -v | --verbose ) [VERBOSE]
                       # --save_file_destination ) [SAVE_FILE_DESTINATION]
                       --num_iterations ) [NUM_ITERATIONS]
                       --setting ) {'regression', 'classification'}
                       --classes ) [CLASSES]
                       --bias ) [BIAS]
                       --lower_bound ) [LOWER_BOUND]
                       --upper_bound ) [UPPER_BOUND]
                       {'GIN'} ...
```

Commented arguments are mostly used for internal development. After all the optional arguments, one must specify the **GNN model** as a positional argument, followed by all its specific arguments detailed in previous sections. For more information on the arguments, types and descriptions, run ```python3 run_teacher.py --help```. 



### Student Setting

---

The student setting ...

```bash
python3 run_model.py --

```

TBD

