from ml_metrics import *
from utils import *
from pyevolve import *
import random
def getBestCP(y,y_pred):
    def score(y,y_pred):
        def fitness_funcion(cps):
            cps = list(cps)
            cps = [ c + random.uniform(0.0000001,0.000000000001)for c in cps]
            cps = np.sort(cps)
            cutpoints = np.concatenate([[-99999999999999999],cps,[999999999999999]])
            y_cp = pd.cut(y_pred,bins=cutpoints,labels=[1,2,3,4,5,6,7,8])           
            score = quadratic_weighted_kappa(y_cp,y)
            return score
        return fitness_funcion
    genome = G1DList.G1DList(7)
    genome.evaluator.set(score(y,y_pred))
    genome.mutator.set(Mutators.G1DListMutatorRealRange)
    genome.setParams(rangemin=0, rangemax=9)
    ga = GSimpleGA.GSimpleGA(genome) 
    ga.setPopulationSize(200)
    ga.setGenerations(42)
    ga.selector.set(Selectors.GRouletteWheel)

    pop = ga.getPopulation() 
    pop.scaleMethod.set(Scaling.SigmaTruncScaling)
    ga.evolve(1)
    cps = list(ga.bestIndividual())
    cps = [ c + random.uniform(0.0000001,0.000000000001)for c in cps]
    cps = np.sort(cps)
    print len(cps)
    return cps
    