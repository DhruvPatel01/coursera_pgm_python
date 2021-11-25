import itertools
import sys

import numpy as np

sys.path.insert(0, '..')
import sol
from commons.factor import Factor


def generate_allele_genotype_mappers(num_alleles):
    alleles_to_genotypes = {}
    genotypes_to_alleles = []
    
    for i in range(num_alleles):
        for j in range(i, num_alleles):
            alleles_to_genotypes[i, j] = alleles_to_genotypes[j, i] = len(genotypes_to_alleles)
            genotypes_to_alleles.append((i, j))
    
    return alleles_to_genotypes, genotypes_to_alleles


def child_copy_given_freqs_factor(allele_freqs, gene_copy_var):
    F = Factor([gene_copy_var], [len(allele_freqs)], init=0.0)
    for i, freq in enumerate(allele_freqs):
        F[i] = freq
    return F


def child_copy_given_parentals_factor(num_alleles, gene_copy_var_child, gene_copy_var1, gene_copy_var2):
    n = num_alleles
    F = Factor([gene_copy_var_child, gene_copy_var1, gene_copy_var2], [n, n, n], init=0.0)
    for i in range(n): #  Father
        for j in range(n): #  Mother
            for k in range(n): #  child
                if i == j == k:
                    F[k, i, j] = 1.0
                elif i == k or j == k:
                    F[k, i, j] = 0.5
    return F



####

net_template = """net 
{
        node_size = (90 36);
}
"""

node_template = """
node {label}
{{
        label = "{label}";
        position = {position};
        states = {states};
}}
"""

potential_template = """
potential ({var} |{given})
{{
        data = {data};
}}
"""

def iter_vals(f):
    domains = reversed(f.domains)
    for assignment in itertools.product(*domains):
        yield f[tuple(reversed(assignment))]

def send_to_samiam(pedigree, factor_list, allele_list, phenotype_list, positions, output_file):
    names = pedigree['names']
    parents = pedigree['parents']
    
    num_peoples = len(names)
    
    with open(output_file, 'w') as f:
        f.write(net_template)
        
        genotypes = ' '.join('"%s%s"' % x for x in itertools.combinations_with_replacement(allele_list, 2))
        genotypes = "(%s)" % genotypes
        
        phenotypes = ' '.join('"%s"' % p for p in phenotype_list)
        phenotypes = "(%s)" % phenotypes
        
        for i in range(num_peoples):
            name = "%sGenotype" % names[i]
            position = "(%d %d)" % (positions[i][0][0], positions[i][0][1])
            f.write(node_template.format(label=name, position=position, states=genotypes))
            
            name = "%sPhenotype" % names[i]
            position = "(%d %d)" % (positions[i][1][0], positions[i][1][1])
            f.write(node_template.format(label=name, position=position, states=phenotypes))
            
        for i in range(num_peoples):
            F = factor_list[i]
            var = "%sGenotype" % names[i]
            if len(F.vars) == 1:
                given = ""
            else:
                j, k = parents[i]
                given = " %sGenotype %sGenotype" % (names[j], names[k])
            data = '(' + ' '.join('%f' % x for x in iter_vals(F)) + ')'
            f.write(potential_template.format(var=var, given=given, data=data))
            
        for i in range(num_peoples):
            F = factor_list[num_peoples+i]
            var = "%sPhenotype" % names[i]
            given = " %sGenotype" % names[i]
            
            data = '('
            it = iter_vals(F)
            max_len = len(F.val)//len(F.domains[0])
            for j in F.domains[0]:
                data += '(' + ' '.join('%f' % next(it) for _ in range(max_len)) + ')\n' + ' '*16
            data = data[:-17] + ')'
            f.write(potential_template.format(var=var, given=given, data=data))
                
            
def send_to_samiam_copy(pedigree, factor_list, allele_list, phenotype_list, positions, output_file):
    names = pedigree['names']
    parents = pedigree['parents']
    
    num_peoples = len(names)
    
    genes = '(' + ' '.join('"%s"' % x for x in allele_list) + ')'
    phenotypes = '(' + ' '.join('"%s"' % p for p in phenotype_list) + ')'
    
    with open(output_file, 'w') as f:
        f.write(net_template)
        
        for i, per_name in enumerate(names):
            name = per_name + 'Parent1GeneCopy'
            position = "(%d %d)" % (positions[i][0][0], positions[i][0][1])
            f.write(node_template.format(label=name, position=position, states=genes))
            
            name = per_name + 'Parent2GeneCopy'
            position = "(%d %d)" % (positions[i][1][0], positions[i][1][1])
            f.write(node_template.format(label=name, position=position, states=genes))
            
            name = "%sPhenotype" % names[i]
            position = "(%d %d)" % (positions[i][2][0], positions[i][2][1])
            f.write(node_template.format(label=name, position=position, states=phenotypes))
            
        for i, per_name in enumerate(names):
            F = factor_list[i]
            if len(F.vars) == 1:
                given = ""
            else:
                p1 = names[parents[i][0]]
                given = " %sParent1GeneCopy %sParent2GeneCopy" % (p1, p1)
            data = '(' + ' '.join('%f' % x for x in iter_vals(F)) + ')'
            f.write(potential_template.format(var=per_name+'Parent1GeneCopy', given=given, data=data))
            
        for i, per_name in enumerate(names):
            F = factor_list[i+num_peoples]
            if len(F.vars) == 1:
                given = ""
            else:
                p1 = names[parents[i][1]]
                given = " %sParent1GeneCopy %sParent2GeneCopy" % (p1, p1)
            data = '(' + ' '.join('%f' % x for x in iter_vals(F)) + ')'
            f.write(potential_template.format(var=per_name+'Parent2GeneCopy', given=given, data=data))
            
        for i, per_name in enumerate(names):
            F = factor_list[i+2*num_peoples]
            given = " %sParent1GeneCopy %sParent2GeneCopy" % (per_name, per_name)
            data = '(' + ' '.join('%f' % x for x in iter_vals(F)) + ')'
            f.write(potential_template.format(var=per_name+'Phenotype', given=given, data=data))
        
