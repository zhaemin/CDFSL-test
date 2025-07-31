

##  How to Run

###  1. Supervised Only

Standard supervised training without memory or distillation.

```bash
bash scripts/sup_only.sh
```

---

###  2. Buffer-based Training

Supervised training enhanced with memory replay (buffer).

```bash
bash scripts/sup_only.sh
```

Modify the following arguments in the script/config:

- **Buffer size**  
  Change `MEM_SIZE` to control the buffer size.  
  Examples:
  - `MEM_SIZE=500`
  - `MEM_SIZE=2000`

- **Distillation loss function**  
  Set `DISTILL_LOSS` to define the distillation method.  


---

###  3. Align (Distillation Loss)

Training with Align, which uses distillation loss without a memory buffer.

```bash
bash scripts/align.sh
```

Modify the following argument:

- **Distillation loss function**  
  Set `DISTILL_LOSS` to control which distillation loss is applied.  

---
