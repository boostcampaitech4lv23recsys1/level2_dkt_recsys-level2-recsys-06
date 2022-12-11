## field-aware factorization machines

### Get started
```
cd ffm
python main.py
```

### parameters
```
# train.py

self.embed_dim = 16
self.epochs = 5
self.learning_rate = 1e-3
self.weight_decay = 1e-5
self.log_interval = 100
self.device = 'cuda'
```

### Performance
| public | private |
| :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------:
|                   0.7491                                |                         0.7414          
