# TODO

- In `src\thalia\regions\prefrontal.py`:
    ```
    # NOTE: Do NOT grow STP here - Prefrontal only has recurrent STP,
    # which tracks n_output (not n_input) and is grown in grow_output() only
    ```
- In `tests\unit\regions\test_thalamus_stp.py`
    ```
    # L6 should show stronger or equal depression (lower or equal ratio)
    # Note: With default config both use STPType.DEPRESSING (U=0.5), so they may be equal
    ```
