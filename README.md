# Accelerating Monte Carlo Dropout-based Bayesian Neural Networks on an FPGA

Bayesian neural networks (BNNs) are making significant progress in many research areas where decision making needs to be accompanied by uncertainty estimation. Being able to quantify uncertainty while making decisions is essential for understanding when the model is over-/under-confident, and hence BNNs are attracting interest in safety-critical applications, such as autonomous driving, robotics or risk quantification. Nevertheless, BNNs have not been as widely used in industrial practice, mainly because of their limiting compute cost. We target the computational limitations of BNNs and present a reconfigurable FPGA-based accelerator for a wide range of convolutional BNN architectures that improves their hardware performance.

- [Accelerating Monte Carlo Dropout-based Bayesian Neural Networks on an FPGA](#accelerating-monte-carlo-dropout-based-bayesian-neural-networks-on-an-fpga)
  - [Structure](#structure)
    - [Requirements & Installation](#requirements--installation)
      - [Requirements](#requirements)
    - [Experiments](#experiments)
      - [Datasets](#datasets)
  - [Authors](#authors)


## Structure

```
.
├── README.md
├── hardware                # All code related to hardware              
                              implementation on an FPGA
├── requirements.txt        
└── software                # All code related to software experiments   
    ├── experiments         # Individual scripts fro running the experiments
```

### Requirements & Installation

#### Requirements

All requirements are in the root of the directory and you can install them recursively as: 

```
pip3 install -r requirements.txt
```

All the hyper-parameters are listed in the source files and we provide dirct scripts that helped us to acquire the results. The hyperparametes were tuned empirically. 

### Experiments

To run the experiments simply navigate to the `software/experiments` folder, select the type of the BNN that you would like to benchmark, navigate to its folder and simply run: 

```

python3 <dataset>.py --gpu <-1, or GPU id>, -seed <1-5>
```

You might find the `quantised` folder for each directory, and these scripts in the underlying folder run scripts for quantisation for th erespective networks and datasets. 

If you wish to run the experiments in batch in the same way how we obtained them you can use scripts again under `software/experiments` folder and run: 

```
./run_experiments.sh <GPU id> 
# or with respect to already run results
./run_samples_experiments.sh <GPU id> <Number of samples>

```

To obtain the data shown in the paper or plots describing the results navigate to the respective folder under `experiments/plots` and run the scripts again with Python 3. 

#### Datasets

Experiemnts were performed on MNIST, CIFAR-10 and SVHN with respect to LeNet=, VGG-11 and ResNet-18 and Monte Carlo Dropout with a dropout rate `0.25` adequately applied.

## Authors

Hongxiang Fan, Martin Ferianc, Miguel Rordrigues, Wayne Luk