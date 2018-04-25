require(metaheuristicOpt)
require(DEoptim)

source('neural.R')

errorRate <- function(cl1, cl2){
  f <- table(cl1 == cl2)
  tr <- f[names(f)=='FALSE'] / length(cl1)
  return (tr)
}

fitnessFunction <- function(wieghts){
  map = mapVectorToNueral(wieghts)
  result <- rnnClassify(map$hiddenWieghts, map$recurrentWieghts, map$outputWieghts, rep(0, hiddenSize))
  
  return(errorRate(result, label))
}


harmonySearch <- function(wieghtsSize){
  numVar <- wieghtsSize
  rangeVar <- matrix(c(-1, 1), nrow=2)
  PAR <- 0.1
  HMCR <- 0.95
  bandwith <- 0.15
  
  ## calculate the optimum solution using Harmony Search algorithm
  resultHS <- HS(fitnessFunction, optimType="MIN", numVar, numPopulation=20, 
                 maxIter=100, rangeVar, PAR, HMCR, bandwith)
  optWieghts_harmonySearch <<- resultHS
  print(fitnessFunction(resultHS))
}

defEvo <- function(wieghtsSize){
  
  upper <- rep(1, wieghtsSize)
  lower <- rep(-1, wieghtsSize)
  dd <- DEoptim(fn=fitnessFunction, lower=lower, upper=upper, 
                DEoptim.control( storepopfrom = 1, itermax = 50, trace=F))
  optWieghts_defEvo <<- dd$optim$bestmem
  print(fitnessFunction(dd$optim$bestmem))
}


prticleSwarmOpt <- function(wieghtsSize){

  Vmax <- 2
  ci <- 1.5
  cg <- 1.5
  w <- 0.7
  numVar <- wieghtsSize
  rangeVar <- matrix(c(-1,1), nrow=2)
  
  ## calculate the optimum solution using Particle Swarm Optimization Algorithm
  resultPSO <- PSO(fitnessFunction, optimType="MIN", numVar, numPopulation=20, 
                   maxIter=50, rangeVar, Vmax, ci, cg, w)
  
  optWieghts_prticleSwarmOpt <<- resultPSO
  
  print(fitnessFunction(resultPSO))
  
}

antLoinOpt <- function(wieghtsSize){
  ## Define parameter 
  numVar <- wieghtsSize
  rangeVar <- matrix(c(-1,1), nrow=2)
  
  ## calculate the optimum solution using Ant Lion Optimizer 
  resultALO <- ALO(fitnessFunction, optimType="MIN", numVar, numPopulation=20, 
                   maxIter=100, rangeVar)
  
  optWieghts <<- resultALO
  
  print(fitnessFunction(resultALO))
}
