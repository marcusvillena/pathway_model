* 0_0.ipynb
    * main structure, pooling classes, encoder/decoder structure, training module etc.
    * mse/mae ~ std in reconstruction; needs improvement

* 0_1.ipynb
    * cleaned up code to ./modules
    * INCOMPLETE; troubleshooting 0_0
    * check: simple decoder (MLP only, pathway unaware) to see if it is the bottleneck
    * otherwise, need to revise encoder structure

* 1_0.ipynb
    * manual transformer attention modules (MHA, MQA, GQA)
    * did not realize nn.MultiheadAttention exists :/

* 1_1.ipynb
    * testing with nn.MultiheadAttention
