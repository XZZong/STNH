# STNH
A Python implementation of the method proposed in "Social Trust Network Embedding with Hash and Graphlet"

# Requirements
The implementation is tested under Python 3.7.6, with the folowing packages installed:
- `networkx==2.5`
- `numpy==1.18.4`
- `scikit-learn==0.23.1`
- `tqdm==4.43.0`

# Examples
Train an STNH model on the deafult `WikiElec` dataset and test it on link prediction task
```python
python main.py
python evaluation.py WikiElec 0.2
```
