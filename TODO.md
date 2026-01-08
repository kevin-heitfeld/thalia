# TODO

- In `src\thalia\regions\prefrontal.py`:
    ```
    # NOTE: Do NOT grow STP here - Prefrontal only has recurrent STP,
    # which tracks n_output (not n_input) and is grown in grow_output() only
    ```
