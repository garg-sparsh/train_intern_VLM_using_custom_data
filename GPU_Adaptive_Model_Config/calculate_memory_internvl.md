## Models

- InternVL2_5-2B-MPO
    ```
    HW: Setup with 1 node, 8 GPUs per node.
    SW: Model with 2205M total params, 189M largest layer params.
    per CPU  |  per GPU |   Options
    55.47GB  |   0.71GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
    98.60GB  |   0.71GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
    49.30GB  |   1.22GB | offload_param=none, offload_optimizer=cpu , zero_init=1
    98.60GB  |   1.22GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    8.47GB   |   5.33GB | offload_param=none, offload_optimizer=none, zero_init=1
    98.60GB  |   5.33GB | offload_param=none, offload_optimizer=none, zero_init=0
    ```

- InternVL2_5-4B-MPO
    ```
    HW: Setup with 1 node, 8 GPUs per node.
    SW: Model with 3712M total params, 310M largest layer params.
    per CPU  |  per GPU |   Options
    93.36GB  |   1.16GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
    165.97GB |   1.16GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
    82.98GB  |   2.02GB | offload_param=none, offload_optimizer=cpu , zero_init=1
    165.97GB |   2.02GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    13.89GB  |   8.94GB | offload_param=none, offload_optimizer=none, zero_init=1
    165.97GB |   8.94GB | offload_param=none, offload_optimizer=none, zero_init=0
    ```

- InternVL2_5-8B-MPO
    ```
    HW: Setup with 1 node, 8 GPUs per node.
    SW: Model with 8075M total params, 379M largest layer params.
    per CPU  |  per GPU |   Options
    203.06GB |   1.41GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
    361.00GB |   1.41GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
    180.50GB |   3.29GB | offload_param=none, offload_optimizer=cpu , zero_init=1
    361.00GB |   3.29GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    16.95GB  |  18.33GB | offload_param=none, offload_optimizer=none, zero_init=1
    361.00GB |  18.33GB | offload_param=none, offload_optimizer=none, zero_init=0
    ```

- InternVL2_5-26B-MPO
    ```
    HW: Setup with 1 node, 8 GPUs per node.
    SW: Model with 25514M total params, 568M largest layer params.
    per CPU   |  per GPU |   Options
    641.57GB  |   2.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
    1140.57GB |   2.12GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
    570.29GB  |   8.06GB | offload_param=none, offload_optimizer=cpu , zero_init=1
    1140.57GB |   8.06GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    25.42GB   |  55.58GB | offload_param=none, offload_optimizer=none, zero_init=1
    1140.57GB |  55.58GB | offload_param=none, offload_optimizer=none, zero_init=0
    ```



## Explanation
- We are taking an example for model `InternVL2_5-2B-MPO`
### Hardware Setup (HW)
```
    1 node, 8 GPUs per node → This setup implies that the training will be distributed across 2 GPUs on a single machine.
    2205M total parameters → The model has 2.2 billion trainable parameters.
    189M largest layer params → The largest single layer of the model has 189 million parameters.
```

### Software Setup (SW)
The table lists different memory consumption scenarios for CPU and GPU based on different parameter offloading and zero initialization strategies.

```
per CPU  |  per GPU |   Options
 55.47GB |   0.71GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
 98.60GB |   0.71GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
 49.30GB |   1.22GB | offload_param=none, offload_optimizer=cpu , zero_init=1
 98.60GB |   1.22GB | offload_param=none, offload_optimizer=cpu , zero_init=0
 8.47GB  |   5.33GB | offload_param=none, offload_optimizer=none, zero_init=1
 98.60GB |   5.33GB | offload_param=none, offload_optimizer=none, zero_init=0

```

#### Key Parameters and Their Effects

- Offloading Strategy
    `offload_param=cpu` → Model parameters are stored on the CPU instead of GPU.
    `offload_optimizer=cpu` → The optimizer state is stored on the CPU instead of GPU.
    `offload_param=none` → Model parameters remain on the GPU.
    `offload_optimizer=none` → Optimizer states remain on the GPU.

- Zero Initialization (zero_init)
    `zero_init=1` → Zero-initialized buffers, which can help save memory.
    `zero_init=0`→ No zero-initialization, meaning more memory is consumed.


#### Observations

- When parameters and optimizer states are offloaded to the CPU (offload_param=cpu, offload_optimizer=cpu)

    - `High CPU usage (~55.47GB) but very low GPU usage (~0.71GB) for zero_init=1`.
    - `High CPU usage (~98.60GB) but very low GPU usage (~0.71GB) for zero_init=0`.
    This is useful if your GPU has limited memory, but training might be slower due to CPU-GPU communication overhead.
    
- When only the optimizer is offloaded to the CPU (offload_param=none, offload_optimizer=cpu)

    - `CPU usage is lower (~49.30GB) but GPU usage increases (~1.22GB) for zero_init=1.`
    - `CPU usage is lower (~98.60GB) but GPU usage increases (~1.22GB) for zero_init=0.`
    A balance between CPU and GPU usage, reducing GPU memory pressure while keeping model parameters in GPU.

- When nothing is offloaded (offload_param=none, offload_optimizer=none)

    - `Low CPU usage of 8.47GB but high GPU usage (~5.53GB) for zero_init=1`
    - `Low CPU usage of 98.60GB but high GPU usage (~5.53GB) for zero_init=0`
    Best for performance if GPU memory is sufficient.


### Conclusion
1. If your GPU has limited memory, use offloading (offload_param=cpu, offload_optimizer=cpu) but expect slower training.
2. If your GPU has sufficient memory, avoid offloading (offload_param=none, offload_optimizer=none) for better speed.
3. If you're optimizing for balanced memory usage, keeping parameters on the GPU (offload_param=none) and offloading the optimizer (offload_optimizer=cpu) can be a good compromise.


