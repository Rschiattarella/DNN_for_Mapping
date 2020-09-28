import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.providers.ibmq import *
from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout



IBMQ.save_account('a4b0b917da2aa2df695ff329567268e62d3862608d4fa50b6cd0ab1781853d3c7616f71d33dc55a33c1ed9af072f7ba6c9dc163400b2a1b54ce720462ad06b78', overwrite=True)
IBMQ.load_account()

def Backend_Topology(name, refresh, show = True, datatime = False):
    '''Funzione che mi restituisce tutte le possibili coppie tra gli slot fisici
    in un dizionario, i cui valori sono una lista contenente error gate ed gate length(ns) del cx tra la coppia,
    i valori sono fissati a  100000, se la coppia di slot non è topologicamente legata
    ______________________________________________________________________________________________________
    poni refresh = True per aggiornare ottenere ad ogni chiamata i dati più aggiornati di calibrazione

    name = Nome del backend
    '''
    Topology_properties = {}
    impossible_length = 100000
    impossible_error = 100000
    provider = IBMQ.get_provider(hub='ibm-q')
    provider.backends(simulator=False)
    backend = provider.get_backend(name)
    #print(backend.configuration().basis_gates)

    if show == True:
        plot_gate_map(backend).show()

    coupling_map = IBMQBackend.configuration(backend).to_dict()['coupling_map']
    if datatime != False:
        prp = backend.properties(datetime=datatime).to_dict()
    else:
        prp = backend.properties(refresh=refresh).to_dict()

    Topology_properties['backend_name'] = prp['backend_name']
    Topology_properties['last_update_date'] = prp['last_update_date']

    gates = prp['gates']
    cx_list = []
    for i in range(len(gates)):
        if gates[i]['gate'] == 'cx':
            cx_list.append([gates[i]])

    coupling = {}
    for i in range(len(prp['qubits'])):
        for j in range(len(prp['qubits'])):
            if i != j:
                string_error = 'edge_error_' + str(i) + str(j)
                string_length = 'edge_length_' + str(i) + str(j)
                if not check_in_list([i, j], coupling_map): #se la lista è vuota
                    coupling[string_error] = impossible_error
                    coupling[string_length] = impossible_length
                else:
                    for ind in check_in_list([i, j], coupling_map):
                        coupling[string_error] = cx_list[ind][0]['parameters'][0]['value']
                        coupling[string_length] = cx_list[ind][0]['parameters'][1]['value']

    Topology_properties['coupling'] = coupling

    return Topology_properties








def Qubit_properties(name, refresh, datatime = False):
    '''funzione che ritorna un dizionario i cui valori sono le calibrazioni di ogni qubit relativo alla chiave
    in questione:
    T1 (µs) = Relaxation Time
    T2 (µs) = Dephasing time
    readout error
    prob_meas0_prep1
    prob_meas1_prep0
    ______________________________________________________________________________________________________
    poni refresh = True per aggiornare ottenere ad ogni chiamata i dati più aggiornati di calibrazione

    name = Nome del backend
    '''
    Qubit_properties = {}
    T1 = []
    T2 = []
    readout_error = []
    prob_meas0_prep1 = []
    prob_meas1_prep0 = []
    provider = IBMQ.get_provider(hub='ibm-q')
    provider.backends(simulator=False)
    backend = provider.get_backend(name)
    if datatime != False:
        prp = backend.properties(datetime=datatime).to_dict()
    else:
        prp = backend.properties(refresh=refresh).to_dict()
    Qubit_properties['backend_name'] = prp['backend_name']
    Qubit_properties['last_update_date'] = prp['last_update_date']


    prp_qubits = prp['qubits']

    for qubit in prp_qubits:
        for dict in qubit:
            if dict['name'] == 'T1':
                T1.append(dict['value'])
            if dict['name'] == 'T2':
                T2.append(dict['value'])
            if dict['name'] == 'readout_error':
                readout_error.append(dict['value'])
            if dict['name'] == 'prob_meas0_prep1':
                prob_meas0_prep1.append(dict['value'])
            if dict['name'] == 'prob_meas1_prep0':
                prob_meas1_prep0.append(dict['value'])

    keys = ['T1', 'T2', 'readout_error', 'prob_meas0_prep1','prob_meas1_prep0']
    values = [T1, T2, readout_error, prob_meas0_prep1, prob_meas1_prep0]
    for i in range(len(keys)):
        Qubit_properties[keys[i]] = values[i]

    # NB -> Rimuovere colonne probabilità FATTO SOLO PER DATASET NN1
    del Qubit_properties['prob_meas1_prep0']
    del Qubit_properties['prob_meas0_prep1']

    return Qubit_properties




def check_in_list(i, list):
    '''funzione che mi dice in che posizione si trovano gli elementi i nella lista list'''
    index = []
    for j in range(len(list)):
        if i == list[j]:
            index.append(j)
        else:
            continue

    return index