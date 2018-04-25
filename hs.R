require(sigmoid)
require(pROC)
require(metaheuristicOpt)
require(DEoptim)


#AUC <- function(class1, class2) multiclass.roc(class1, class2)$auc[1]

errorRate <- function(cl1, cl2){
  f <- table(cl1 == cl2)
  tr <- f[names(f)=='FALSE'] / length(cl1)
  return (tr)
} 

initializeVars <- function(dataFile, lableCol = 1, hSize = 10){
  rawData <<- read.csv(dataFile)
  
  dataset <<- rawData[-lableCol]
  label <<- rawData[lableCol]
  label <<- as.numeric(t(label))
  
  inputSize <<- length(dataset)
  outSize <<- ceiling(log2(length(table(label))))
  
  hiddenSize <<- hSize
  
}



classify <- function(hiddenWieghts, recurrentWieghts, outputWieghts, recurrent){
  result <- c()
  
  for(i in 1:length(dataset[,1])){
    
    sample = as.double(dataset[i,])
    
    inputResult <- sample %*% hiddenWieghts + recurrent %*% recurrentWieghts
    
    inputResult <- sigmoid(inputResult)
    
    recurrent <- inputResult
    
    output <- inputResult %*% outputWieghts
    output <- ifelse(output < 0.5, 0, 1)
    
    output <-  sum(2^(0:(outSize-1)) * output)
    result <- c(result, output[1])
  }
  return(result)
}


fitnessFunction <- function(wieghts){
  s = 1
  e = inputSize * hiddenSize
  hiddenWieghts <- matrix(wieghts[s:e], 
                          ncol = hiddenSize, 
                          byrow = TRUE)
  
  s = e + 1
  e = e + hiddenSize * hiddenSize
  
  recurrentWieghts <- matrix(wieghts[s:e], 
                             ncol = hiddenSize,
                             byrow = TRUE)
  
  s = e + 1
  e = e + outSize * hiddenSize
  outputWieghts <- matrix(wieghts[s:e], 
                          ncol = outSize, 
                          byrow = TRUE)
  result <- classify(hiddenWieghts, recurrentWieghts, outputWieghts, rep(0, hiddenSize))
  
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
  
  print(fitnessFunction(resultHS))
}

defEvo <- function(wieghtsSize){
  upper <- rep(1, wieghtsSize)
  lower <- rep(-1, wieghtsSize)
  dd = DEoptim(fn=fitnessFunction, lower=lower, upper=upper, 
               DEoptim.control( storepopfrom = 1, itermax = 100, trace=F))
  
  print(fitnessFunction(dd$optim$bestmem))
}
main <- function(){
  file <- "C://Users/polla/Desktop/iris.csv" #file.choose()
  initializeVars(file, lableCol = 5)
  wieghtsSize = inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize * outSize
  
  #harmonySearch(wieghtsSize)
  defEvo(wieghtsSize)
  
}

main()