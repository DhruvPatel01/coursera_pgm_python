import pickle
import json
from collections import OrderedDict
import numpy as np
import os

import requests
from scipy.sparse import coo_matrix


class SubmissionBase:
    """
    Author: Gerges Dib (https://github.com/dibgerge)
    This is adapted from Gerges Dib's repository for ML assignments to work with PGM assignments.
    """
    submit_url = 'https://www.coursera.org/api/onDemandProgrammingScriptSubmissionsController.v1'
    save_file = 'token.pkl'

    def __init__(self, assignment_name, assignment_key, part_names):
        self.assignment_name = assignment_name
        self.assignment_key = assignment_key
        self.part_names = part_names
        self.login = None
        self.token = None
        self.functions = OrderedDict()
        self.args = dict()

    def grade(self):
        print('\nSubmitting Solutions | Programming Exercise %s\n' % self.assignment_name)
        self.login_prompt()

        # Evaluate the different parts of exercise
        parts = OrderedDict()
        for part_id, result in self:
            if not isinstance(result, str):
                output = sprintf('%0.5f ', result)
            else:
                output = result.strip()
            #print(part_id, output)
            parts[str(part_id)] = {'output': output}
        ret = self.request(parts)
        if 'errorMessage' in ret:
            print(ret['errorMessage'])
        else:
            print(ret)
            print("Submitted successfully, view results on assignment page.")

    def login_prompt(self):
        if os.path.isfile(self.save_file):
            with open(self.save_file, 'rb') as f:
                login, token = pickle.load(f)
            reenter = input('Use token from last successful submission (%s)? (Y/n/N): '
                            '\n Y: use both previous email and token'
                            '\n n: use only email'
                            '\n N: reenter email and token: ' % (login, ))

            self.login = None
            if reenter == '' or reenter[0] == 'Y' or reenter[0] == 'y':
                self.login, self.token = login, token
                return
            elif reenter == 'n':
                self.login = login
                os.remove(self.save_file)
            else:
                os.remove(self.save_file)
        
        if not self.login:
            self.login = input('Login (email address): ')
        self.token = input('Token: ')

        # Save the entered credentials
        if not os.path.isfile(self.save_file):
            with open(self.save_file, 'wb') as f:
                pickle.dump((self.login, self.token), f)

    def request(self, parts):
        params = {
            'assignmentKey': self.assignment_key,
            'secret': self.token,
            'parts': parts,
            'submitterEmail': self.login}
        with open('/tmp/python_post', 'w') as f:
            f.write(json.dumps(params))
        req = requests.post(self.submit_url, data={'jsonBody': json.dumps(params)})
        return req.json()

    def __iter__(self):
        for part_id in self.functions:
            yield part_id

    def __setitem__(self, key, value):
        self.functions[key] = value


def sprintf(fmt, arg):
    """ Emulates (part of) Octave sprintf function. """
    if isinstance(arg, tuple):
        # for multiple return values, only use the first one
        arg = arg[0]

    if isinstance(arg, (np.ndarray, list)):
        # concatenates all elements, column by column
        return ' '.join(fmt % e for e in np.asarray(arg).ravel('F'))
    else:
        return fmt % arg


def adj_matrix_to_adj_list(matrix):
    edges = {}
    for i in range(len(matrix)):
        nbs = set()
        for j in range(len(matrix)):
            if matrix[i, j] == 1:
                nbs.add(j)
        edges[i] = nbs
    return edges


def adj_list_to_csgraph(adjlist):
    coo = []
    for u in adjlist:
        for v in adjlist[u]:
            coo.append((u, v, 1))
    coo = coo_matrix(coo)
    return coo.tocsr()
