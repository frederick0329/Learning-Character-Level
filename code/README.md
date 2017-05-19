# code

The foloder contains the code for running the experiments

## Usage

Training

For text
```
th doall.lua -data_path [preprocessed folder] -model_path [model_folder_path] -gpu_id 1
```

For image
```
th doall.lua -data_path [preprocessed folder] -model_path [model_folder_path] -img -gpu_id 1
```

Testing
```
th eval.lua -data_path [preprocessed folder] -model_path [model_path] -gpu_id 1 -batch_size n
```

For image
```
th eval.lua -data_path [preprocessed folder] -model_path [model_path] -img -gpu_id 1 -batch_size n 
```

