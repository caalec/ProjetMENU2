import matplotlib.pyplot as plt
import numpy as np


#Variables globales
c = 1.5 #Vitesse du son dans l'eau (mm/µs)
a = 0.5 #Rayon transducteur (mm)
z = 10 #Hauteur du point d'observation (mm)
N_r  = 15 #Nombre de transducteurs 2*N_r+1
n = 100 #Nombre de points
r = np.linspace(0.001,10, n) #Liste des r étudiés entre 0 et 5mm
t = np.linspace(0.95*z/c,2*np.sqrt(z**2+a**2)/c, n) #Liste temporelle des points entre 95%z/c et 2*sqrt((z^2+a^2)/c)
f_e = 1/(t[1]-t[0]) #Fréquence d'échantillonage
r_s = 5.5 #Point de focalisation





#Fonctions 

def h(a,r,z,t):
    
    if r<=a:
        
        if c*t < z:
            return(0)
        elif z<c*t and c*t<np.sqrt(z**2+(a-r)**2):
            return(1)
        elif np.sqrt(z**2+(a-r)**2)<c*t and c*t<np.sqrt(z**2+(a+r)**2):
            return(1/np.pi*np.arccos((r**2+c**2*t**2-z**2-a**2)/(2*r*np.sqrt(c**2*t**2-z**2))))
        else:
            return(0)
    else:
        if c*t<np.sqrt(z**2+(a-r)**2):
            return(0)
        elif np.sqrt(z**2+(a-r)**2)<c*t and c*t<np.sqrt(z**2+(a+r)**2):
            return(1/np.pi*np.arccos((r**2+c**2*t**2-z**2-a**2)/(2*r*np.sqrt(c**2*t**2-z**2))))
        else:
            return(0)

def h_list_c(a,r,z,t):
    """
    Renvoie la liste des h pour un r et un t donné
    """

    h_list = np.ones((len(r),len(t)))

    for i in range(len(r)):
        for j in range(len(t)):
            h_list[i,j] = h(a, r[i], z, t[j])

    return(h_list)


def dh_c(h_list,pas):
    """
    Calcul de la dérivée
    """
    return np.diff(h_list)/pas


def e(f_c, b_w, N):
    """
    Signal d'entrée impulsionnel de fréquence centrale f_c et de bande passante b_w (en %) de N points
    """
    t = np.linspace(-(N-1)/2, (N-1)/2, N)/f_e
    alpha = np.pi/np.log(2)*(b_w*f_c/2)**2
    S = np.sin(2*np.pi*f_c*t)*np.exp(-alpha*t**2)
    return(S)


def convolution(e, dh, h_list):
    """
    Calcul de la convolution entre e et dh
    """
    E = np.fft.fft(e[:-1])
    H = np.fft.fft(dh)
    res = np.ones(np.shape(H),dtype=np.complex128)
    for i in range(len(h_list[0])):
        res[i]=np.fft.ifft(E*H[i])
    return(res)


def pression(a,r,z,t,signal_e):
    h_list = h_list_c(a,r,z,t)
    return(np.real(convolution(signal_e, dh_c(h_list,t[1]-t[0]), h_list)))


def transducteur_reception(r_s):
    """
    Calcule le signal reçu par l'émission d'un point situé en r_s pour chaque transducteur
    """
    Signaux = np.ones((2*N_r+1,len(t)-1))
    j = 0
    p_list = pression(a,r,z,t,e(2,1,len(t)))
    for i in range(-N_r,N_r+1):
        distance = abs(r_s - 2*i*a)
        Signaux[j] = p_list[int(round(n*distance/21))]
        j += 1
    return(Signaux)


def retournement_temporel(Signaux):
    """
    Retourne le signal temporellement dans le temps
    """
    return(np.concatenate((np.flip(Signaux, axis=1),np.zeros((2*N_r+1,1))),axis=1))


def cut_useless_signal(Signaux):
    """
    Retire toutes les données inférieures à 1e-1 au début d'un signal
    """
    j_max = len(Signaux[0])
    for i in range(len(Signaux)):
        for j in range(len(Signaux[0])):
            if Signaux[i,j] > 1e-1 and j<j_max:
                j_max=j
    Signaux_resize = Signaux[:,j_max:]
    Signaux_resize = np.concatenate((Signaux_resize, np.zeros((2*N_r+1, j_max))), axis = 1)
    return(Signaux_resize)


def champ_focalise(a,r,z,t,r_s):

    p_list_f = np.zeros((len(r),len(t)-1))

    Signaux = cut_useless_signal(retournement_temporel(transducteur_reception(r_s)))

    for i in range(-N_r,N_r+1):
        p_list_f += pression(a,np.abs(r+i*a),z,t, Signaux[i])

    return(p_list_f)


fig, ax = plt.subplots()

C = ax.pcolormesh(t[:-1], r, champ_focalise(a,r,z,t, r_s), cmap='jet')
ax.set_xlabel('t')
ax.set_ylabel('r')

fig.colorbar(C, ax=ax)

plt.show()
