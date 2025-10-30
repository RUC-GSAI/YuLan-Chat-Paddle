from x2paddle.convert import pytorch2paddle
pytorch2paddle(module=torch_module,
                        save_dir="./pd_model",
                        jit_type="trace",
                        input_examples=[torch_input])