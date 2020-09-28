from Circuit_features import *
from Backend_features import *
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import Unroller
from qiskit.circuit.random import random_circuit
import pandas as pd
import random
import os.path
from qiskit.transpiler.passes import LookaheadSwap, StochasticSwap
from qiskit.transpiler import CouplingMap



def pick_label(circ, backend, coupling_map, optimization_level, show=False):
    '''
    Funzione che restituisce il dizionario con il mapping, scegliendo come label layout quello che minimizza la depth
    del circuito tra dense layout e noise_adaptive sommata alla depth delle operazioni dopo il routing.
    In questa maniera tengo anche conto di qual'è il layout che permette di minimizzare le operazioni di swap
    '''
    new_circ_lv3 = transpile(circ, backend=backend, optimization_level=optimization_level)
    new_circ_lv3_na = transpile(circ, backend=backend, optimization_level=optimization_level,layout_method='noise_adaptive')
    #plot_circuit_layout(new_circ_lv3_na, backend).show()
    #plot_circuit_layout(new_circ_lv3, backend).show()
    cp = CouplingMap(couplinglist=coupling_map)
    depths = []
    for qc in [new_circ_lv3_na, new_circ_lv3]:
        depth = qc.depth()
        pass_manager = PassManager(LookaheadSwap(coupling_map=cp))
        lc_qc = pass_manager.run(qc)
        pass_manager = PassManager(StochasticSwap(coupling_map=cp))
        st_qc = pass_manager.run(qc)
        depths.append(depth + lc_qc.depth())
        depths.append(depth + st_qc.depth())
        #print('depth=', depth, ' depth + routing_lc_qc= ', depth + lc_qc.depth(), ' depth + routing_st_qc=',depth + st_qc.depth())

    if depths.index(min(depths)) < 2:
        print('na')
        if show == True:
            plot_circuit_layout(new_circ_lv3_na, backend).show()
        return new_circ_lv3_na._layout.get_physical_bits()

    if depths.index(min(depths)) >= 2:
        print('not na')
        if show == True:
            plot_circuit_layout(new_circ_lv3, backend).show()
        return new_circ_lv3._layout.get_physical_bits()



def add_line(circuit, backend_name, refresh=True, show=True, optimization_level=3, datatime=False):
    '''
    Funzione che mi costruisce righe del dataset ritornandomi la fatta da una prima componente che è
    una lista contenente i titoli delle features, e una seconda lista contenente i valori corrispondenti.
    _____________________________________________________________________________________________________
    I Titoli dei label sono le ultime features della riga: il titolo è il nome della slot fisica, il valore
    è il nome del qubit virtuale
    '''

    provider = IBMQ.get_provider(hub='ibm-q')
    provider.backends(simulator=False)
    backend = provider.get_backend(backend_name)
    basis_gats = backend.configuration().basis_gates
    pass_ = Unroller(basis_gats)
    pm = PassManager(pass_)
    new_circ = pm.run(circuit)

    size_backend = len(backend.properties(refresh=refresh).to_dict()['qubits'])

    CA = circuit_analysis(circuit, size_backend=size_backend, show=show)
    #print(CA)
    BT = Backend_Topology(backend_name, refresh, show, datatime=datatime)
    #print(BT)
    QP = Qubit_properties(backend_name, refresh, datatime=datatime)
    #print(QP)

    coupling_map = IBMQBackend.configuration(backend).to_dict()['coupling_map']

    label = pick_label(new_circ, backend=backend, coupling_map=coupling_map, optimization_level=optimization_level, show = show)
    #print(label)


    Title_names =['last_update_date', 'backend_name'] + list(CA.keys())[:3] + list(CA['cx'].keys()) + list(CA['measure'].keys()) + list(BT['coupling'].keys())
    for i in range(size_backend):
        for title in list(QP.keys())[2:]:
            Title_names.append(title+'_'+str(i))

    Title_names = Title_names + list(range(size_backend))


    date_name = [BT['last_update_date'], BT['backend_name']]
    Values = date_name + list(CA.values())[:3] + list(CA['cx'].values()) + list(CA['measure'].values()) + list(BT['coupling'].values())
    for i in range(size_backend):
        for value in list(QP.keys())[2:]:
            Values.append(QP[value][i])

    for qubit in range(size_backend):
        #print(label[qubit].register.name)
        if label[qubit].register.name == 'q':
            Values.append(label[qubit].index)
        else:
            Values.append(None)



    return [Title_names,Values]




def update_csv(file_name, backend_name, rows_to_add, random_n_qubit=5, random_depth=10, show = False, min_n_qubit=1, datatime=False):

    '''
    Funzione che aggiunge la riga al dataset contenuto in file_name.
    __________________________________________________________________________________
    backend_name = backend su cui fare il mapping
    rows_to_add = numero di righe da aggiungere variano random il numero di qubit e la depth del circuito
    random_n_qubit = estremo superiore di randint (default 5)
    random_depth = estremo superiore di randint per la depth (default 10)
    '''
    if os.path.exists(file_name) == True:
        df = pd.read_csv(file_name)
        start = len(list(df['Index']))

    else:
        start = 0
    print(start)
    for j in range(start, start + rows_to_add):
        print(j, backend_name, j-start+1)
        n_qubit = random.randint(min_n_qubit, random_n_qubit)
        depth = random.randint(1, random_depth)
        try:
            circ = random_circuit(n_qubit, depth, measure=True)
            #circ.draw(output='mpl').show()
            l = add_line(circ,backend_name, optimization_level=3, refresh=True, show= show, datatime=datatime)

        except qiskit.transpiler.exceptions.TranspilerError :
            print('Impossibile mappare il circuito: Generando nuovo qc')
            error = 1
            while error==1:
                try:
                    circ = random_circuit(n_qubit, depth, measure=True)
                    # circ.draw(output='mpl').show()
                    l = add_line(circ, backend_name, optimization_level=3, refresh=True, show=show)
                    error = 0
                except qiskit.transpiler.exceptions.TranspilerError:
                    continue

        d={}

        for i in range(len(l[0])):
            d[str(l[0][i])] = l[1][i]

        #print(d)

        df = pd.DataFrame(d, index=[j])
        if j==0:
            df.to_csv(file_name, mode='a')
        else:
            df.to_csv(file_name, mode='a',header=None)


import datetime
iteration = [1]
for it in range(iteration[0]):
    month = random.randint(1,5)
    day = random.randint(1,29)
    data = datetime.date(2020, month, day)
    n_qs = [2,3,4,5]
    #backend_name = ['ibmq_vigo', 'ibmq_ourense',  'ibmq_rome', 'ibmq_essex','ibmq_burlington']
    backend_name_1 = ['ibmq_burlington']
    file_name = '/home/rschiattarella/dataset/dataset_tesi/Dataset_Prova_4_08.csv'

    for backend in backend_name_1:
        for n_q in n_qs:
            update_csv(file_name, backend, rows_to_add=1, random_n_qubit=5, random_depth=2, min_n_qubit=5, datatime=data, show=True)