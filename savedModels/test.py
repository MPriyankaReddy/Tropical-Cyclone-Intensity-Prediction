import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print(model.predict({'la':10.4,'lon':52,'ecp':982,'mssw':65,'pd':22,'sst':26.1}))