DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS vehicles CASCADE;
DROP TABLE IF EXISTS places CASCADE;
DROP TABLE IF EXISTS caracteristics CASCADE;
DROP TABLE IF EXISTS holidays CASCADE;

-- Création de la table principale
CREATE TABLE caracteristics (
    Num_Acc TEXT,
    an TEXT,
    mois TEXT,
    jour TEXT,
    hrmn TEXT,
    lum TEXT,
    agg TEXT, 
    "int" TEXT,
    atm TEXT,
    col TEXT,
    com TEXT,
    adr TEXT,
    gps TEXT,
    lat TEXT,
    long TEXT,
    dep TEXT,
    PRIMARY KEY (Num_Acc)
);

CREATE TABLE places (
    Num_Acc TEXT,
    catr TEXT,
    voie TEXT,
    v1 TEXT,
    v2 TEXT,
    circ TEXT,
    nbv TEXT,
    pr TEXT,
    pr1 TEXT,
    vosp TEXT,
    prof TEXT,
    plan TEXT,
    lartpc TEXT,
    larrout TEXT,
    surf TEXT,
    infra TEXT,
    situ TEXT,
    env1 TEXT,
    PRIMARY KEY (Num_Acc),
    FOREIGN KEY (Num_Acc)
        REFERENCES caracteristics(Num_Acc)
        ON DELETE CASCADE
);

CREATE TABLE vehicles(
    Num_Acc TEXT,
    senc TEXT,
    catv TEXT,
    occutc TEXT,
    obs TEXT,
    obsm TEXT,
    choc TEXT,
    manv TEXT,
    num_veh TEXT,
    PRIMARY KEY (Num_Acc, num_veh), -- Un accident peut avoir plusieurs vehicules. Besoin de créer une clé composite
    FOREIGN KEY (Num_Acc)
        REFERENCES caracteristics(Num_Acc)
        ON DELETE CASCADE
);

CREATE TABLE users (
    Num_Acc TEXT,
    place TEXT,
    catu TEXT,
    grav TEXT,
    sexe TEXT,
    trajet TEXT,
    secu TEXT,
    locp TEXT,
    actp TEXT,
    etatp TEXT,
    an_nais TEXT,
    num_veh TEXT,
    PRIMARY KEY (Num_Acc, num_veh, place), -- Un accident peut avoir plusieurs passagers dans un même véhicule. Besoin de créer une clé composite unique
    FOREIGN KEY (Num_Acc, num_veh)
        REFERENCES vehicles(Num_Acc, num_veh)
        ON DELETE CASCADE
);

CREATE TABLE holidays(
    ds TEXT,
    holiday TEXT,
    PRIMARY KEY (ds)
);
   
    