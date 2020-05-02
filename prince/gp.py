import numpy as np
import scipy.io as scio

def random_walk_with_restart(network, initial_value, alpha=0.1, epoch=50):
    N = network.shape[0]
    R = initial_value / np.sum(initial_value)
    Y = R
    for i in range(epoch):
        R = (1 - alpha) * Y + alpha * np.dot(network, R)
    res = np.argsort(-R)[:10] + 1
    return res

if __name__ == '__main__':
    g_p_network = np.array(scio.loadmat('g_p_network.mat')['g_p_network'])
    ppi_network = np.array(scio.loadmat('ppi_network.mat')['ppi_network'])
    phenotype_network = np.array(scio.loadmat('phenotype_network.mat')['phenotype_network'])
    ppi_network = np.divide(ppi_network, np.sum(ppi_network, axis=0))
    phenotype_id = int(input('Please input phenotype ID'))
    #phenotype_id = 100800
    index = int(np.argwhere(phenotype_network[..., 0] == phenotype_id)[0][0])
    query_phenotype = np.argsort(-phenotype_network[index][1:])[:5]
    #print(query_phenotype)
    result = random_walk_with_restart(ppi_network, np.sum(g_p_network[..., query_phenotype], axis=1), 0.1, 50)
    print("Related genes(top10):",result)