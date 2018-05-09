require(metaheuristicOpt)
require(DEoptim)

source('neural.R')

iterations <- 100
population <- 50

errorRate <- function(cl1, cl2){
  f <- table(cl1 == cl2)
  tr <- f[names(f)=='FALSE'] / length(cl1)
  return (tr)
}

fitnessFunctionRnn <- function(wieghts){
  map = mapVectorForRnn(wieghts)
  result <- rnnClassify(map$hiddenWieghts, map$recurrentWieghts, map$outputWieghts, rep(0, hiddenSize))
  
  return(errorRate(result, label))
}

fitnessFunctionLstm <- function(wieghts){
  map = mapVectorForLstm(wieghts)
    
  result <- lstmClassify(map$Wi, map$Ui, map$Wf, map$Uf, map$Wo, map$Uo, map$Wc, map$Uc, map$outputWieghts)
  
  return(errorRate(result, label))
}

harmonySearch <- function(wieghtsSize, nural = "lstm"){
  numVar <- wieghtsSize
  rangeVar <- matrix(c(-1, 1), nrow=2)
  PAR <- 0.1
  HMCR <- 0.95
  bandwith <- 0.15
  fitnessFunction <- ""
  if(nural == "lstm") 
    fitnessFunction <- fitnessFunctionLstm
  else
    fitnessFunction <- fitnessFunctionRnn
  
  ## calculate the optimum solution using Harmony Search algorithm
  resultHS <- HS(fitnessFunction, optimType="MIN", numVar, numPopulation=population, 
                 maxIter=iterations, rangeVar, PAR, HMCR, bandwith)
  optWieghts_harmonySearch <<- resultHS
  print(fitnessFunction(resultHS))
  return(resultHS)
  
}

defEvo <- function(wieghtsSize, nural = "lstm"){
  
  upper <- rep(1, wieghtsSize)
  lower <- rep(-1, wieghtsSize)
  fitnessFunction <- ""
  if(nural == "lstm") 
    fitnessFunction <- fitnessFunctionLstm
  else
    fitnessFunction <- fitnessFunctionRnn
  
  dd <- DEoptim(fn=fitnessFunction, lower=lower, upper=upper, 
                DEoptim.control( storepopfrom = 1, itermax = iterations, trace=F))
  optWieghts_defEvo <<- dd$optim$bestmem
  print(fitnessFunction(dd$optim$bestmem))
  
  return(dd$optim$bestmem)
  
}


prticleSwarmOpt <- function(wieghtsSize, nural = "lstm"){

  Vmax <- 2
  ci <- 1.5
  cg <- 1.5
  w <- 0.7
  numVar <- wieghtsSize
  rangeVar <- matrix(c(-1,1), nrow=2)
  fitnessFunction <- ""
  if(nural == "lstm") 
    fitnessFunction <- fitnessFunctionLstm
  else
    fitnessFunction <- fitnessFunctionRnn
  
  
  ## calculate the optimum solution using Particle Swarm Optimization Algorithm
  resultPSO <- PSO(fitnessFunction, optimType="MIN", numVar, numPopulation=population, 
                   maxIter=iterations, rangeVar, Vmax, ci, cg, w)
  
  optWieghts_prticleSwarmOpt <<- resultPSO
  
  print(fitnessFunction(resultPSO))
  return(resultPSO)
  
}

antLoinOpt <- function(wieghtsSize, nural = "lstm"){
  ## Define parameter 
  numVar <- wieghtsSize
  rangeVar <- matrix(c(-1,1), nrow=2)
  
  fitnessFunction <- ""
    if(nural == "lstm") 
      fitnessFunction <- fitnessFunctionLstm
  else
    fitnessFunction <- fitnessFunctionRnn
  
  ## calculate the optimum solution using Ant Lion Optimizer 
  resultALO <- ALO(fitnessFunction, optimType="MIN", numVar, numPopulation=population, 
                   maxIter=iterations, rangeVar)
  
  optWieghts <<- resultALO
  
  print(fitnessFunction(resultALO))
  return(optWieghts)
  
}


