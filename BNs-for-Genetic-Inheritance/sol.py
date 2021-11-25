import sys

import numpy as np

sys.path.insert(0, '..')
import helper
from commons.factor import Factor


def phenotype_given_genotype_mendelian_factor(is_dominant, genotype_var, phenotype_var):
    """
    This function computes the probability of each phenotype given the
    different genotypes for a trait.  It assumes that there is 1 dominant
    allele and 1 recessive allele.

    If you do not have much background in genetics, you should read the
    on-line Appendix or watch the Khan Academy Introduction to Heredity Video
    (http://www.khanacademy.org/video/introduction-to-heredity?playlist=Biology)
    before you start coding.

    For the genotypes, assignment 0 maps to homozygous dominant, assignment 1
    maps to heterozygous, and assignment 2 maps to homozygous recessive.  For
    example, say that there is a gene with two alleles, dominant allele A and
    recessive allele a.  Assignment 0 would map to AA, assignment 1 would
    make to Aa, and assignment 2 would map to aa.  For the phenotypes, 
    assignment 0 maps to having the physical trait, and assignment 1 maps to 
    not having the physical trait.

    THE VARIABLE TO THE LEFT OF THE CONDITIONING BAR MUST BE THE FIRST
    VARIABLE IN THE .vars FIELD FOR GRADING PURPOSES

    Args:
      is_dominant: 1 if the trait is caused by the dominant allele (trait 
      would be caused by A in example above) and 0 if the trait is caused by
      the recessive allele (trait would be caused by a in the example above)
      
      genotype_var: The variable number for the genotype variable (goes in the
      .vars part of the factor)
      
      phenotype_var: The variable number for the phenotype variable (goes in
      the .vars part of the factor)

    Returns:
      phenotype_factor: Factor denoting the probability of having each 
      phenotype for each genotype
    """

    phenotype_factor = Factor([phenotype_var, genotype_var], [2, 3], init=0.0)
    
    # Solution Start
    
    
    
    # Solution End
    
    return phenotype_factor


def phenotype_given_genotype(alpha_list, genotype_var, phenotype_var):
    """
    This function computes the probability of each phenotype given the 
    different genotypes for a trait. Genotypes (assignments to the genotype
    variable) are indexed from 0 to the number of genotypes(exclusive)
    , and the alphas are provided in the same order as the corresponding
    genotypes so that the alpha for genotype assignment i is alpha_list[i].
    
    For the phenotypes, assignment 0 maps to having the physical trait, and 
    assignment 1 maps to not having the physical trait.
    
    THE VARIABLE TO THE LEFT OF THE CONDITIONING BAR MUST BE THE FIRST
    VARIABLE IN THE .vars FIELD FOR GRADING PURPOSES
    
    Args:
        alpha_list: Vector of alpha values for each genotype (n,) vector,
        where n is the number of genotypes) -- the alpha value for a genotype
        is the probability that a person with that genotype will have the
        physical trait 
        
        genotype_var: The variable number for the genotype variable (goes in the
        .vars part of the factor)
        
        phenotype_var: The variable number for the phenotype variable (goes in
        the .vars part of the factor)
    
    Returns:
        phenotype_factor: Factor in which the val has the probability of having 
        each phenotype for each genotype combination (note that this is the 
        FULL CPD with no evidence observed)
    """
    
    n = len(alpha_list)
    phenotype_factor = Factor([phenotype_var, genotype_var], [2, n], init=0.0)
    
    # Solution Start
    
   
    
    # Solution End
    
    return phenotype_factor


def genotype_given_allele_freqs_factor(allele_freqs, genotype_var):
    """
    This function computes the probability of each genotype given the allele 
    frequencies in the population.

    Note that we assume that the copies of the gene are independent.  Thus,
    knowing the allele for one copy of the gene does not affect the
    probability of having each allele for the other copy of the gene.  As a
    result, the probability of a genotype is the product of the frequencies 
    of its constituent alleles (or twice that product for heterozygous 
    genotypes).

    Args:
        allele_freqs: An n x 1 vector of the frequencies of the alleles in the 
            population, where n is the number of alleles

        genotype_var: The variable number for the genotype (goes in the .vars
        part of the factor)

    Returns:
        genotype_factor: Factor in which the val has the probability of having 
            each genotype (note that this is the FULL CPD with no evidence 
            observed)

    The number of genotypes is (number of alleles choose 2) + number of 
    alleles -- need to add number of alleles at the end to account for 
    homozygotes
    """
    
    n = num_alleles = len(allele_freqs)
    card = (n * (n-1))//2 + n
    genotype_factor = Factor([genotype_var, ], [card, ], init=0.0)

    """
    Each allele has an ID that is the index of its allele frequency in the 
    allele frequency list. Each genotype also has an ID. We need allele and
    genotype IDs so that we know what genotype and alleles correspond to each
    probability in the the factor. For example, F[0] corresponds to the 
    probability of having the genotype with genotype ID 0, which consists 
    of having two copies of the allele with allele ID 0. 
    
    There is a mapping from a pair of allele IDs to genotype 
    IDs and from genotype IDs to a pair of allele IDs below; we compute this 
    mapping using helper.generate_allele_genotype_mappers(num_alleles). 
    (A genotype consists of 2 alleles.)
    """
    
    alleles_to_genotypes, genotypes_to_alleles = helper.generate_allele_genotype_mappers(n)
    
    """
    One or both of these matrices might be useful.

    1.  alleles_to_genotypes: A dict that maps pairs of allele IDs to 
    genotype IDs. If alleles_to_genotypes[i, j] = k, then the genotype 
    with ID k comprises of the alleles with IDs i and j

    2.  genotypes_to_alleles: A list of length m of allele IDs, where m is the 
    number of genotypes -- if genotypes_to_alleles[k] = (i, j), then the 
    genotype with ID k is comprised of the allele with ID i and the allele 
    with ID j
    """
    
    # Solution Start
    
    
        
    # Solution End
    
    return genotype_factor


def genotype_given_parents_genotypes_factor(num_alleles, 
                                            genotype_var_child, 
                                            genotype_var_parent_one, genotype_var_parent_two):
    """
    This function computes a factor representing the CPD for the genotype of
    a child given the parents' genotypes.

    THE VARIABLE TO THE LEFT OF THE CONDITIONING BAR MUST BE THE FIRST
    VARIABLE IN THE .vars FIELD FOR GRADING PURPOSES

    When writing this function, make sure to consider all possible genotypes 
    from both parents and all possible genotypes for the child.

    Args:
      num_alleles: int that is the number of alleles
      
      genotype_var_child: Variable number corresponding to the variable for the
      child's genotype (goes in the .vars part of the factor)
      
      genotype_var_parent_one: Variable number corresponding to the variable for
      the first parent's genotype (goes in the .vars part of the factor)
      
      genotype_var_parent_two: Variable number corresponding to the variable for
      the second parent's genotype (goes in the .vars part of the factor)

    Returns:
      genotype_factor: Factor in which val is probability of the child having 
      each genotype (note that this is the FULL CPD with no evidence 
      observed)

    The number of genotypes is (number of alleles choose 2) + number of 
    alleles -- need to add number of alleles at the end to account for homozygotes
    """
    
    n = num_alleles
    genotype_factor = Factor([], [])
    alleles_to_genotypes, genotypes_to_alleles = helper.generate_allele_genotype_mappers(n)
    
    # Solution Start
    
    
    # Solution End
    
    return genotype_factor


def construct_genetic_network(pedigree, allele_freqs, alpha_list):
    """
    This function constructs a Bayesian network for genetic inheritance.  It
    assumes that there are only 2 phenotypes.  It also assumes that either 
    both parents are specified or neither parent is specified.

    In Python, each variable will have a number.  We need a consistent way of
    numbering variables when instantiating CPDs so that we know what
    variables are involved in each CPD.  For example, if IraGenotype is in
    multiple CPDs and IraGenotype is number variable 1 in a CPD, then
    IraGenotype should be numbered as variable 1 in all CPDs and no other
    variables should be numbered 1; thus, every time variable 1 appears in
    our network, we will know that it refers to Ira's genotype.

    Here is how the variables should be numbered, for a pedigree with n
    people:

    1.  The first n variables should be the genotype variables and should
    be numbered according to the index of the corresponding person in
    pedigree['names']; the ith person with name pedigree['names'][i] has genotype
    variable number i. i starts from 0.

    2.  The next n variables should be the phenotype variables and should be
    numbered according to the index of the corresponding person in
    pedigree['names']; the ith person with name pedigree['names'][i] has phenotype
    variable number n+i. i starts from 0.

    Here is an example of how the variable numbering should work: if
    pedigree['names'] = ['Ira', 'James', 'Robin'] and
    pedigree['parents'] = [None, (1, 3), None]
    then the variable numbering is as follows:

    Variable 0: IraGenotype
    Variable 1: JamesGenotype
    Variable 2: RobinGenotype
    Variable 3: IraPhenotype
    Variable 4: JamesPhenotype
    Variable 5: RobinPhenotype

    Input:
      pedigree: Dict that includes names and parent-child
      relationships. It has following keys,
          'names': A list of size n containing name of people.
          'parents': A list of size n containing parents. Each
              element is either None (in case parents are unknown) or
              a tuple of two integers.
      
      allele_freqs: Frequencies of alleles in the population
      
      alpha_list: A list of size m of alpha values for genotypes, where m is the
      number of genotypes -- the alpha value for a genotype is the 
      probability that a person with that genotype will have the
      physical trait

    Returns:
      factor_list: A list  of factors for the genetic network.
    """
    
    # Initialize factors
    # The number of factors is twice the number of people because there is a 
    # factor for each person's genotype and a separate factor for each person's 
    # phenotype.  Note that the order of the factors in the list does not
    # matter.
    num_peoples = len(pedigree['names'])
    factor_list = [Factor([], []) for _ in range(2*num_peoples)]
    
    # Solution Start
   


    # Solution End
    
    return factor_list


def phenotype_given_copies_factor(alpha_list, num_alleles, gene_copy_var1, gene_copy_var2, phenotype_var):
    """
    This function makes a factor whose values are the probabilities of 
    a phenotype given an allele combination. Note that a person has one
    copy of the gene from each parent.

    In the factor, each assignment maps to the allele at the corresponding
    location on the allele list, so allele assignment 0 maps to
    allele_list[0], allele assignment 1 maps to allele_list[1], ....  For the
    phenotypes, assignment 0 maps to having the physical trait, and
    assignment 1 maps to not having the physical trait.

    THE VARIABLE TO THE LEFT OF THE CONDITIONING BAR MUST BE THE FIRST
    VARIABLE IN THE .vars FIELD FOR GRADING PURPOSES

    Args:
        alpha_list: list of length m of alpha values for the different genotypes,
        where m is the number of genotypes -- the alpha value for a genotype
        is the probability that a person with that genotype will have the
        physical trait
        
        num_alleles: int that is the number of alleles
        
        gene_copy_var1: Variable number corresponding to the variable for
        the first copy of the gene (goes in the .vars part of the factor)
        
        gene_copy_var2: Variable number corresponding to the variable for
        the second copy of the gene (goes in the .vars part of the factor)
        
        phenotype_var: Variable number corresponding to the variable for the 
        phenotype (goes in the .vars part of the factor)

    Returns:
        phenotype_factor: Factor in which the values are the probabilities of 
        having each phenotype for each allele combination (note that this is 
        the FULL CPD with no evidence observed)
    """
    
    phenotype_factor = Factor([], [])
    
    alleles_to_genotypes, genotypes_to_alleles = helper.generate_allele_genotype_mappers(num_alleles)
    
    # Solution Start
    
    
    
    # Solution End
    
    return phenotype_factor
    
    
def construct_decoupled_genetic_network(pedigree, allele_freqs, alpha_list):
    """
    This function constructs a Bayesian network for genetic inheritance.  It
    assumes that there are only 2 phenotypes.  It also assumes that, in the
    pedigree, either both parents are specified or neither parent is
    specified.

    In Python, each variable will have a number.  We need a consistent way of
    numbering variables when instantiating CPDs so that we know what
    variables are involved in each CPD.  For example, if IraGenotype is in
    multiple CPDs and IraGenotype is number variable 0 in a CPD, then
    IraGenotype should be numbered as variable 0 in all CPDs and no other
    variables should be numbered 0; thus, every time variable 0 appears in
    our network, we will know that it refers to Ira's genotype.

    Here is how the variables should be numbered, for a pedigree with n
    people:

    1.  The first n variables should be the gene copy 1 variables and should
    be numbered according to the index of the corresponding person in
    pedigree['names']; the ith person with name pedigree['names'][i] has gene copy
    1 variable number i.  If the parents are specified, then gene copy 1 is the
    copy inherited from pedigree['names'][pedigree['parents'][i, 0]].

    2.  The next n variables should be the gene copy 2 variables and should
    be numbered according to the index of the corresponding person in
    pedigree['names']; the ith person with name pedigree['names'][i] has gene copy
    2 variable number n+i.  If the parents are specified, then gene copy 2 is the
    copy inherited from pedigree['parents'][i, 1]).

    3.  The final n variables should be the phenotype variables and should be
    numbered according to the index of the corresponding person in
    pedigree['names']; the ith person with name pedigree['names'][i] has phenotype
    variable number 2n+i.

    Here is an example of how the variable numbering should work: if
    pedigree['names'] = ['Ira', 'James', 'Robin'] and
    pedigree['parents'] = [None, (1, 3), None], then
    the variable numbering is as follows:

    Variable 0: IraGeneCopy1
    Variable 1: JamesGeneCopy1
    Variable 2: RobinGeneCopy1
    Variable 3: IraGeneCopy2
    Variable 4: JamesGeneCopy2
    Variable 5: RobinGeneCopy2
    Variable 6: IraPhenotype
    Variable 7: JamesPhenotype
    Variable 8: RobinPhenotype

    Args:
      pedigree: Dict that includes the names and parents of each person
      
      alleleFreqs: List of length n of allele frequencies in the population,
          where n is the number of alleles
      
      alphaList: List of length m of alphas for different genotypes, where m is
          the number of genotypes -- the alpha value for a genotype is the 
          probability that a person with that genotype will have the physical 
          trait

    Returns:
      factor_list: List of Factors for the genetic network.
    """
    
    num_people = len(pedigree['names'])
    factor_list = [Factor([], []) for i in range(3*num_people)]
    parents = pedigree['parents']
    
    # Hint: You can use helper.child_copy_given_freqs_factor and
    # helper.child_copy_given_parentals_factor
    
    # Solution Start
    
    
    
    # Solution End
    
    return factor_list


def construct_sigmoid_phenotype_factor(allele_weights, gene_copy_var1_list, 
                                       gene_copy_var2_list, phenotype_var):
    """
    This function takes A list of of lists(weights) of alleles' weights 
    and constructs a factor expressing a sigmoid CPD.
    
    You can assume that there are only 2 genes involved in the CPD.
    
    In the factor, for each gene, each allele assignment maps to the allele
    whose weight is at the corresponding location.  For example, for gene 0,
    allele assignment 0 maps to the allele whose weight is at
    allele_weights[0][0] (same as w_1^1), allele assignment 1 maps to the
    allele whose weight is at allele_weights[0][1](same as w_2^1),....  

    You may assume that there are 2 possible phenotypes.
    For the phenotypes, assignment 0 maps to having the physical trait, and
    assignment 1 maps to not having the physical trait.
    
    THE VARIABLE TO THE LEFT OF THE CONDITIONING BAR MUST BE THE FIRST
    VARIABLE IN THE .vars FIELD FOR GRADING PURPOSES
    
    Input:
      allele_weights: A list of of weights, where each row is an list
          of weights for the alleles for a gene
      
      gene_copy_var1_list: list of size m (m is the number of genes) of variable 
          numbers that are the variable numbers for each of the first parent's 
          copy of each gene (numbers in this list go in the .vars part of the
          factor)

      gene_copy_var2_list: list of size m (m is the number of genes) of variable 
          numbers that are the variable numbers for each of the second parent's 
          copy of each gene (numbers in this list go in the .vars part of the
          factor) -- Note that both copies of each gene are from the same person,
          but each copy originally came from a different parent
      
      phenotype_var: Variable number corresponding to the variable for the 
          phenotype (goes in the .vars part of the factor)
   
    Output:
      phenotype_factor: Factor in which the values are the probabilities of 
      having each phenotype for each allele combination (note that this is 
      the FULL CPD with no evidence observed)
    """
    phenotype_factor = Factor([], [])
    
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    # Solution Start
    
                    
    
    # Solution End
    
    return phenotype_factor