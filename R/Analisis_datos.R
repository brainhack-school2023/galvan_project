########################################
## PROYECTO: 
## OBJETIVO: Encontrar la función de distribución de densidad y sus parámetros
########################################

Función de distribución empírica
---------------------
Dada una muestra aleatoria de tamaño n de una distribución de probabilidad P,
la función de distribución empírica Pn se define como la distribución que asigna probabilidad 1/n 
a cada valor xi con i=1,2,...,n.
 En otras palabras, Pn asigna a un conjunto A en el espacio muestral de x la probabilidad empírica:
Pn(A)=#{xi pertence A}/n

Parámetros y estadísticas
---------------------
Un parámetro es una función de la distribución de probabilidad θ=t(P),
mientras que una estadística es una función de la muestra x

El principio del plug-in es un método para estimar parámetros a partir de muestras; 
la estimación plug-in de un parámetro θ=t(P) se define como:
^θ=t(Pn)

El principio del plug-in provee de una estimación más no habla de precisión: 
usaremos el bootstrap para estudiar el sesgo y el error estándar del estimador plug-in

¿Qué tan bien funciona el principio del plug-in?
Suele ser muy bueno cuando la única información disponible de P es la muestra x
bajo esta circunstancia no puede ser superado como estimador de titon, 
al menos no en el sentido asintótico de teoría estadística (cuando n tiende a infinito)
 https://tereom.github.io/est-computacional-2018

La distribución muestral de una estadística es la distribución de probabilidad de la misma, 
considerada como una variable aleatoria.

¿Por qué bootstrap?
En el caso de la media ^θ = ^x la aplicación del principio del plug-in para el cálculo de errores estándar es inmediata; 
sin embargo, hay estadísticas para las cuáles no es fácil aplicar este método.
 
El método de aproximarlo con simulación, en la práctica no podemos seleccionar un número arbitrario de
muestras de la población, sino que tenemos únicamente una muestra.

La idea del bootstrap es replicar el método de simulación para aproximar el error estándar, 
esto es seleccionar muchas muestras y calcular la estadística de interés en cada una, 
con la diferencia que las muestras se seleccionan de la distribución empírica a falta 
de la distribución poblacional.


###

  
library(ggplot2)
library(gridExtra)
library(MASS)

col1<-"firebrick"
col2<-"dodgerblue3"
col3<-"goldenrod1"
col4<-"darkolivegreen4"
col5<-"darkorange1"
col6<-"chocolate"
col7<-"darkblue"
col8<-"forestgreen"

rm(list=ls())
setwd("C:/hcgalvan/Repositorios/hcgalvan_project/data/union/End")
ls()
#datos=scan(file="seleccionestudio.csv",sep=",")
temp = gsub(".*target.*", "", readLines("seleccionestudio.csv"))
data<-read.table(text=temp, sep=",", header=TRUE)
names(data)
sl2l_dm <- as.numeric(unlist(data['sl2l_diameter']))
datos<-data.frame("sl2l_dm"= sl2l_dm)
datos

#########################
# https://rpubs.com/hllinas/R_Filtrar_DataFrames
#---Represento graficamente la distribucion acumulada de sl2l_dm en control y estudio--
par(mfrow=c(1,2))
Fn1<-ecdf(as.numeric(unlist(subset(data, label==1, select=c(sl2l_diameter)))))
Fn2<-ecdf(as.numeric(unlist(subset(data, label==0, select=c(sl2l_diameter)))))
plot(Fn2, main=" ecdf(x)", col="blue")
lines(Fn1, col="red")
#---Represente graficamente la distrib acumulada de ambos juntos
ECDF<-ecdf(sl2l_dm)
plot(ECDF,col="red",lwd=3,xlab="sl2l_dm",ylab="")

#########################
# AHORA Queremos estimar la función de densidad
En vez de aproximar a traves de la empírica, voy a realizar a traves de densidad, 
quiero aproximar a una funcion.
ninguna suposición de forma (normal, weibul..), solo que es f es suave

################
Pensemos ahora que esta familia no hacemos supuesto de forma (no relacionamos a una e...), si suponemos que tiene esperanza
y varianza. Esta cuenta de los momentos es valida para este criterio.
Estamos estimando 

###
Ahora pensemos lo otro: tenemos resultado pero no la fuente, queremos inferior la fuente
Estimador de Maxima Verosimilitud
Elejir el valor del parametro que maximiza la probabilidad de que observe lo que salio.

density(sl2l_dm, from=25, to=25, n=1, kernel="rectangular", bw=5)$y
# Bloxplot
par(mfrow=c(1,2))
boxplot(sl2l_dm, vertical = TRUE)
hist(as.numeric(unlist(datos)), col="deepskyblue")

summary(sl2l_dm)

# estimacion de parametros

est_mm<-4*
est_mv
# Histogramas
#1
par(mfrow=c(2,2))
hist(as.numeric(unlist(datos)), col="deepskyblue")
hist(sl2l_dm, col="deepskyblue",breaks =seq(5,8,10))
hist(sl2l_dm, col="deepskyblue",breaks =seq(10,16,10))
hist(sl2l_dm, col="deepskyblue",breaks =seq(16,18,10))

#2
par(mfrow=c(1,1))
hist(as.numeric(unlist(datos)), col="deepskyblue",breaks =c(12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32))
par(mfrow=c(1,1))

# con ggplot2
g1<-ggplot(datos, aes(X1)) + 
  geom_histogram(aes(x=sl2l_dm,y = after_stat(density)),binwidth = 2,alpha=0.5, 
                 fill = "mediumpurple3", color = "black",
                 breaks =seq(10,25,2)) + 
  theme_light()

g2<-ggplot(datos, aes(X1)) + 
  geom_histogram(aes(x=sl2l_dm,y = after_stat(density)),binwidth = 2,alpha=0.5, 
                 fill = "mediumpurple3", color = "black",
                 breaks =seq(12,28,2)) + 
  theme_light()

g3<-ggplot(datos, aes(X1)) + 
  geom_histogram(aes(x=sl2l_dm,y = after_stat(density)),binwidth = 2,alpha=0.5, 
                 fill = "mediumpurple3", color = "black",
                 breaks =seq(12,32,2)) + 
  theme_light()

g4<-ggplot(datos, aes(X1)) + 
  geom_histogram(aes(x=sl2l_dm,y = after_stat(density)),binwidth = 2,alpha=0.5, 
                 fill = "mediumpurple3", color = "black",
                 breaks =seq(12,34,2)) + 
  theme_light()

grid.arrange(g1, g2, g3, g4, nrow = 2)

#3 Estimemos la probabilidad

estimamos<-function(x,h,Datos)
{
  p_est<-sum(Datos<=(x+h) & Datos>=(x-h))/length(Datos)
  return(p_est)
}
x<-sort(sl2l_dm) #los ordeno solo porque cuando los grafico quiero que me funcione bien el comando lines

h<-2
est1<-c()
for(i in 1:length(sl2l_dm))
{
  est1[i]<-estimamos(x[i],h,sl2l_dm)
  
}

h<-4
est2<-c()
for(i in 1:length(sl2l_dm))
{
  est2[i]<-estimamos(x[i],h,sl2l_dm)
  
}

h<-6
est3<-c()
for(i in 1:length(sl2l_dm))
{
  est3[i]<-estimamos(x[i],h,sl2l_dm)
  
}
est3
est2
est1
col1<-"firebrick"
col2<-"dodgerblue3"
col3<-"goldenrod1"
#plot(x, est3, col=col)
plot(x,est3, col=col3, type="l", lwd=2)
lines(x,est2, col=col2, type="l", lwd=2)
lines(x,est1, col=col1, type="l", lwd=2)


estimamos(23,4,sl2l_dm)

# Estimadores basados en nucleos

# Estimacion Parzen
uniforme<-function(u)
{
  ifelse(u>-1 & u<1,1,0)/2  # es mi indicadora y mi calcula el nucleo uniforme
}


#para nucleos
f_sombrero<-function(x,k,datos,h) #datos= Xi
{
  s<-0
  for(i in 1:length(datos))
  {
    c<-k((x-datos[i])/h)
    s<-s+c
  }
  f<-s/(length(datos)*h)
  return(f)
}  


densidad.est.parzen<-function(x,h,z) # x: datos, z:valor donde exaluo la f
{
  f_sombrero(z,uniforme,x,h)
}


nuevos<-seq(10,32,length=96) #los numeros, son los mínimos y maximos de la muestra
h<-2
f_estimada1<-densidad.est.parzen(datos$sl2l_dm,h,nuevos)
f_est<-data.frame("x"=nuevos,  "estimada1"=f_estimada1)
f_est
densidad.est.parzen(sl2l_dm,2, sl2l_dm[1])

ggplot(f_est,aes(x=x,y=estimada1))+
  geom_line(col="steelblue")+
  theme_light()


h<-4
f_est$estimada2<-densidad.est.parzen(datos$sl2l_dm,h,nuevos)

h<-6
f_est$estimada3<-densidad.est.parzen(datos$sl2l_dm,h,nuevos)


ggplot(f_est)+
  geom_histogram(data=datos,aes(x=sl2l_dm,y = ..density..),binwidth = 2,alpha=0.3, 
                 fill = "mediumpurple3", color = "black",
                 breaks =seq(10,32,2)) + 
  geom_line(aes(x=x,y=estimada1),col="steelblue",lwd=1.5)+
  geom_line(aes(x=x,y=estimada2),col="firebrick",lwd=1.5)+
  geom_line(aes(x=x,y=estimada3),col="olivedrab4",lwd=1.5)+
  theme_light()



#Con density. Density evalua sobre una grilla equiespaciada
# h<-hsil
# la función density es la estimación
hist(datos$sl2l_dm, freq = FALSE,ylim=c(0,0.2))
lines(density(datos$sl2l_dm,kernel = "gaussian",width=h),col="yellowgreen",lwd=2)
lines(density(datos$sl2l_dm,kernel = "epanechnikov",width=h),col="firebrick",lwd=2)
lines(density(datos$sl2l_dm,kernel = "rectangular",width=h),col="steelblue",lwd=2)

# Uso mi función que da lo mismo que el density, pero la puedo evaluar en los puntos que yo quiero.
gauss<-function(u)
{
  k<-exp(-(u^2)/2)/sqrt(2*pi)
  return(k)
}

# Otros nucleos
epa<-function(u)
{
  ifelse(abs(u) < 1,3/4*(1-u^2),0)
} 

estimada5<-f_sombrero(sort(datos$sl2l_dm),gauss,datos$sl2l_dm,2)
estimada5

#ventana de silverman...

h_sil1<-round((4*sd(datos$sl2l_dm)^5/3*length(datos$sl2l_dm))^(1/5),0)
h_sil2<-1.06*min(sd(datos$sl2l_dm),IQR(datos$sl2l_dm)/1.349)*length(datos$sl2l_dm)^(1/5)
h_sil2
h_sil1
sd(datos$sl2l_dm) # Desviación estandar
IQR(datos$sl2l_dm)/1.349 # Desviación estandar - error estandar coincidente con distribucíón normal.

 h<-h_sil1
# la función density es la estimación
hist(datos$sl2l_dm, freq = FALSE,ylim=c(0,0.2))
lines(density(datos$sl2l_dm,kernel = "gaussian",width=h),col="yellowgreen",lwd=2)
lines(density(datos$sl2l_dm,kernel = "epanechnikov",width=h),col="firebrick",lwd=2)
lines(density(datos$sl2l_dm,kernel = "rectangular",width=h),col="steelblue",lwd=2)

h<-h_sil2
# la función density es la estimación
hist(datos$sl2l_dm, freq = FALSE,ylim=c(0,0.2))
lines(density(datos$sl2l_dm,kernel = "gaussian",width=h),col="yellowgreen",lwd=2)
lines(density(datos$sl2l_dm,kernel = "epanechnikov",width=h),col="firebrick",lwd=2)
lines(density(datos$sl2l_dm,kernel = "rectangular",width=h),col="steelblue",lwd=2)


#da re grande

#con otra funcion
bw.nrd(datos$sl2l_dm)
bw.nrd0(datos$sl2l_dm)

#ventana de CV (Convalidaci´on Cruzada por M´axima Verosimilitud)
# primero veamos como funciona el calculo del segundo termino

i<-1
h<-5
f.hat<-c()
for(i in 1:length(datos$sl2l_dm))
{
  f.hat[i]<-f_sombrero(datos$sl2l_dm[i],epa,datos=datos$sl2l_dm[-i],h)
}

# para un h fijo, el segundo termino de LSCV(h)
2*mean(f.hat)

# ventana con LSCV (leave-one-out-Cross-Validation)

lscv<-function(datos,k, h)
{
  f.hat.x<-c()
  for(i in 1:length(datos))
  {
    f.hat.x[i]<-f_sombrero(datos[i],k,datos[-i],h)
  }
  LSCV<-integrate(Vectorize(f_sombrero,"x"),lower=-Inf,upper=Inf,k=k,datos=datos,h=h)$value-2*mean(f.hat.x)
  return(LSCV)
}

h<-seq(12,32,1) # si la ventana es muy chiquita se queja...se queja por todo el integrate...

LSCV_h<-c()
for(j in 1:length(h))
{
  LSCV_h[j]<-lscv(datos$sl2l_dm,epa,h[j])
}

plot(h,LSCV_h,type="l") #pasa que para muchas ventanas no encontró datos...no las tengo en cuenta
plot(h[2:6],LSCV_h[2:6],type = "l")
h[which.max(LSCV_h)] # si tan solo funcionara...




