MIGLIORIE GENERICHE:
- [ ] Separare l'hand_mapper node dal nodo di controllo della mano. Il primo diventa in C++; il secondo, per il momento, data la libreria per il controllo, rimane in Python. Unisci anche il secondo all'Headless_driver_l per stabilire la comunicazione con il device
- [ ] Togli il visualizzatore dall'hand_landmarker e fai un nodo a parte che legge l'immagine, i landmark, ed il feedback di forza, per colorare i link in base al feedback
- [ ] Associa ai landmark una posizione 3D nello spazio

MIGLIORIE CONTROLLO MANO
- [ ] Feedback tattile
  
MIGLIORIE DELL'ALGORITMO DI MAPPING

- [ ] versione matematica più precisa usando i quaternioni?
- [ ] mapping cinematico più avanzato
- [ ] Codice avanzato (punto 6): IK + retargeting avanzato
- [ ] Closed- [ ]loop con force feedback?
- [ ] Teleoperazione con smooth retargeting
- [ ] Sinergie cinematiche?
- [ ] Versione industriale fault- [ ]tolerant
 
Se vuoi spingere al massimo:

- [ ] Impedance control
- [ ] Model-based grasp stabilization
- [ ] Slip detection
- [ ] MPC grasp control
- [ ] Learning-based synergy tuning
- [ ] Adaptive grip strength
