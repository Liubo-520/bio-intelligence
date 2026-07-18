# Davis ABL1 nominal-variant input audit

The project loaders map `target_id` directly to the sequence in `BioInteract/data/raw/davis/target_sequences.csv`. The target-name field is not a model feature. The table below therefore audits the actual sequence inputs.

- Nominal ABL1 entries: 15
- Unique sequence SHA-256 values: 1
- Result: every listed nominal ABL1 entry has the same 1,167-residue input sequence as the `ABL1` record.

| Target ID | Nominal target name | Length | SHA-256 | Identical to ABL1 input? |
|---|---|---:|---|---|
| T0001 | ABL1(E255K) | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0002 | ABL1(F317I) | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0003 | ABL1(F317I)p | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0004 | ABL1(F317L) | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0005 | ABL1(F317L)p | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0006 | ABL1(H396P) | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0007 | ABL1(H396P)p | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0008 | ABL1(M351T) | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0009 | ABL1(Q252H) | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0010 | ABL1(Q252H)p | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0011 | ABL1(T315I) | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0012 | ABL1(T315I)p | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0013 | ABL1(Y253F) | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0014 | ABL1 | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |
| T0015 | ABL1p | 1167 | `cd6a78b82fede215cdb80a797f228098a41178fdde0753728f3781e6221fb1ee` | Yes |

## D0010/D0017 Davis records

These are dataset labels/affinities; they do not make the identical sequence inputs mutation-specific.

| Compound ID | Compound name | Target ID | Nominal target name | Label | Affinity (nM) | pKd |
|---|---:|---|---|---:|---:|---:|
| D0010 | 5328940 | T0001 | ABL1(E255K) | 1 | 0.047 | 9.83268266525 |
| D0010 | 5328940 | T0002 | ABL1(F317I) | 1 | 0.63 | 9.13667713988 |
| D0010 | 5328940 | T0003 | ABL1(F317I)p | 1 | 0.18 | 9.55284196866 |
| D0010 | 5328940 | T0004 | ABL1(F317L) | 1 | 0.11 | 9.67778070527 |
| D0010 | 5328940 | T0005 | ABL1(F317L)p | 1 | 0.029 | 9.8894102897 |
| D0010 | 5328940 | T0006 | ABL1(H396P) | 1 | 0.062 | 9.79048498546 |
| D0010 | 5328940 | T0007 | ABL1(H396P)p | 1 | 0.057 | 9.80410034759 |
| D0010 | 5328940 | T0008 | ABL1(M351T) | 1 | 0.037 | 9.86327943284 |
| D0010 | 5328940 | T0009 | ABL1(Q252H) | 1 | 0.086 | 9.73048705578 |
| D0010 | 5328940 | T0010 | ABL1(Q252H)p | 1 | 0.039 | 9.85698519975 |
| D0010 | 5328940 | T0011 | ABL1(T315I) | 1 | 21 | 7.6757175447 |
| D0010 | 5328940 | T0012 | ABL1(T315I)p | 1 | 3.6 | 8.43179827593 |
| D0010 | 5328940 | T0013 | ABL1(Y253F) | 1 | 0.036 | 9.86646109163 |
| D0010 | 5328940 | T0014 | ABL1 | 1 | 0.12 | 9.65757731918 |
| D0010 | 5328940 | T0015 | ABL1p | 1 | 0.057 | 9.80410034759 |
| D0017 | 3062316 | T0001 | ABL1(E255K) | 1 | 0.047 | 9.83268266525 |
| D0017 | 3062316 | T0002 | ABL1(F317I) | 1 | 0.1 | 9.69897000434 |
| D0017 | 3062316 | T0003 | ABL1(F317I)p | 1 | 0.041 | 9.85078088734 |
| D0017 | 3062316 | T0004 | ABL1(F317L) | 1 | 0.032 | 9.87942606879 |
| D0017 | 3062316 | T0005 | ABL1(F317L)p | 1 | 0.019 | 9.92445303861 |
| D0017 | 3062316 | T0006 | ABL1(H396P) | 1 | 0.025 | 9.90308998699 |
| D0017 | 3062316 | T0007 | ABL1(H396P)p | 1 | 0.046 | 9.83564714422 |
| D0017 | 3062316 | T0008 | ABL1(M351T) | 1 | 0.016 | 9.93554201077 |
| D0017 | 3062316 | T0009 | ABL1(Q252H) | 1 | 0.037 | 9.86327943284 |
| D0017 | 3062316 | T0010 | ABL1(Q252H)p | 1 | 0.064 | 9.78515615195 |
| D0017 | 3062316 | T0011 | ABL1(T315I) | 0 | 890 | 6.05056119896 |
| D0017 | 3062316 | T0012 | ABL1(T315I)p | 0 | 120 | 6.9204569926 |
| D0017 | 3062316 | T0013 | ABL1(Y253F) | 1 | 0.058 | 9.80134291305 |
| D0017 | 3062316 | T0014 | ABL1 | 1 | 0.029 | 9.8894102897 |
| D0017 | 3062316 | T0015 | ABL1p | 1 | 0.046 | 9.83564714422 |
