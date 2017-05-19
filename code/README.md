# code

The foloder contains the code for running the experiments

## Usage

Training

For text
```
th doall.lua -data_path [preprocessed folder] -model_folder [model_folder_path]
```

For image
```
th doall.lua -data_path [preprocessed folder] -model_folder [model_folder_path] -img
```

Testing
```
th eval.lua -data_path [preprocessed folder] -model_path [model_path] -gpu_id 1 -batch_size n
```

For image
```
th eval.lua -data_path [preprocessed folder] -model_path [model_path] -img -gpu_id 1 -batch_size n 
```

