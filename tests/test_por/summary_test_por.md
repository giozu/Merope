# Summary: test_por/ Scripts

## Scopo
Gli script in `test_por/` sono utilizzati per studiare la conducibilità termica effettiva (K_eff) in funzione di:
- **Porosità totale** (P)
- **Spessore grain boundary layer** (delta)
- **Morfologia della porosità** (interconnessa vs distribuita)

## Script Analizzati

### `2_rad_mixed_gen.py` ✅
**Eseguito con successo** - genera belle microstrutture in poco tempo.

**Caratteristiche:**
- **Tre tipi di porosità**:
  1. **Inter-granular pores**: sfere (R=0.03, phi=0.4) distribuite tra i grani
  2. **Intra-granular pores**: due popolazioni di sfere piccole dentro i grani
     - R=0.05, phi=0.03
     - R=0.3, phi=0.19
  3. **Grain boundary layer**: spessore delta=0.003

- **Pattern di overlay (chiave per capire la fisica!)**:
  ```python
  # Primo overlay: inter-pores su grains+boundaries
  dictionnaire1 = {2:0, 3:0}  # Pori inter e boundaries → grains (solid)
  structure3 = merope.Structure_3D(multiInclusions2, multiInclusions, dictionnaire1)

  # Secondo overlay: intra-pores su tutto
  dictionnaire = {0:2}  # Dove ci sono intra-pores, diventano pori
  structure = merope.Structure_3D(structure3, structure4, dictionnaire)
  ```

- **Parametri di fit** (linee 69-71):
  ```python
  alpha = -0.5484
  beta = 1.9214
  gamma = 0.3777
  # Formula: porosity ≈ alpha*inclPhi + beta*delta + gamma
  ```
  Questi parametri legano la porosità totale ai parametri geometrici (inclPhi, delta).

- **Fasi finali**:
  - Phase 0: grains (solid, K=1.0)
  - Phase 1: intra-pores (low K=1e-3)
  - Phase 2: inter-pores (low K=1e-3)

### `iter_delta_IGB_calc.py` ✅
**Eseguito con successo** - sweep K_eff vs delta a porosità ~costante (~0.20).

**Cosa fa:**
1. Genera un policristallo Laguerre (lagR=3, RSA) con grain boundary layer di spessore delta
2. Distribuisce sfere (R=0.3, BOOL) nell'intero volume
3. Usa `merope.Structure_3D(sphIncl, polyCrystal, dictionnaire)` per confinare i pori:
   - `{incl_phase:grains_phase, delta_phase:grains_phase}` = i pori dentro i bordi grano diventano fase 0 (poro)
4. Voxellizza con Average+Voigt, lancia Amitex, salva K_eff

**Parametri:**
- L=10, n3D=100, inclR=0.3, lagR=3 (grani), K=[1.0, 1.0, 1e-3]
- 21 punti: delta da 0.394 a 3.0
- Target porosità: `porosity = 0.2` (riga 66)
- `inclPhi` pre-calcolati (array `a` calibrato a mano, riga 68) per mantenere porosità ~0.20 al variare di delta
- **Nessun loop iterativo** di correzione: i valori di `a` decrescono con delta (da 0.421 a 0.099) perché strati più spessi catturano più pori. La calibrazione non è perfetta, soprattutto a delta piccoli (P reale = 0.26 vs target 0.20)

**Risultati (P~0.20):**
| delta | K_mean | Nota |
|-------|--------|------|
| 0.39  | 0.289  | Interconnessa, K bassa |
| 0.92  | 0.675  | Transizione |
| 1.57  | 0.703  | Quasi plateau |
| 3.00  | 0.712  | Distribuita, K alta |

**Osservazione chiave:** la porosità cala leggermente (0.26 → 0.20) con delta crescente perché a delta piccolo i pori si ammassano nei bordi sottili e si sovrappongono di piu, generando piu volume poroso effettivo.

### `IGB_generator.py`
**Singola run** con lo stesso core Merope di `iter_delta_IGB_calc.py` (solo inter-granular pores, no intra). Parametri ora allineati al primo caso dello sweep: inclR=0.3, inclPhi=0.842, lagR=3, delta=0.394, n3D=100. ThermalAmitex commentato. Ha anche parametri di fit alpha/beta/gamma definiti ma non usati.

### `IGB_porosity_calc.py` ✅
**Eseguito con successo** - sweep K_eff vs porosità a delta fisso. Stesso core Merope di `iter_delta_IGB_calc.py`. Probabilmente usato per generare il grafico K_eff vs Porosity della slide (due curve: delta=L_grain e delta=L_grain/3).

**Cosa fa ora:**
- Sweep di inclPhi: `np.linspace(0.02, 0.3, 20)` → 20 punti di porosità crescente
- delta=3, lagR=3 (quindi delta=L_grain), n3D=100
- Per ogni inclPhi: genera struttura, lancia Amitex, salva K_eff in `aggregated_results.txt`

### `sph_incl_conduct_calc.py` ✅
**Eseguito con successo** - caso base: sfere RSA isolate in matrice, senza grain boundaries ne overlay.

**Cosa fa:**
- Genera sfere RSA (R=0.8, fase 1) in matrice (fase 0), L=20, n3D=40
- Sweep su 5 porosità da 0.05 a 0.30, lancia Amitex per ognuna
- Usa vecchia API `merope.Voxellation_3D`, solo 2 fasi (K=[1.0, 1e-3])

**Risultati:**
| Porosity | K_mean |
|----------|--------|
| 0.05     | 0.925  |
| 0.11     | 0.835  |
| 0.18     | 0.749  |
| 0.24     | 0.667  |
| 0.30     | 0.589  |

K_eff cala quasi linearmente con la porosità — caso classico di inclusioni sferiche distribuite (confrontabile con Maxwell-Garnett).

### `mixed_Intra_inter_calc.py`
**Sweep 2D sistematico** — il modello di microstruttura più completo tra gli script in `test_por/`.

**Cosa fa:**
- Sweep su 10 valori di `inclPhi` (0.02→0.20) × 10 valori di `incl2Phi` (0.01→0.031) = **100 run Amitex**
- Separa esplicitamente **inter-granular pores** (sfere BOOL, R=0.3, `inclPhi`) e **intra-granular pores** (sfere RSA, R=0.05, `incl2Phi`)
- Doppio overlay:
  1. `{2:0, 3:0}`: confina i pori inter nei grain boundaries
  2. `{0:2}`: distribuisce i pori intra dentro i grani
- delta=3, lagR=3 (fissi), L=10, n3D=100

**Perché è il più rappresentativo:**
- Unico script che fa uno **sweep parametrico separato** su inter e intra porosity
- Permette di isolare l'effetto di ciascuna popolazione di pori su K_eff
- Stesso pattern di overlay di `2_rad_mixed_gen.py` ma con esplorazione sistematica dello spazio dei parametri

**Limiti:** delta è fisso (=lagR), quindi non studia l'effetto della morfologia interconnessa vs distribuita. Per quello serve combinare con lo sweep su delta di `iter_delta_IGB_calc.py`.

### `Gauss_multi_rad_gen.py`
Come `2_rad_mixed_gen.py` ma con **distribuzione gaussiana** dei raggi per i pori intra-granulari (N(0.3, 0.13), 5 popolazioni). n3D=300, delta=0.003. ThermalAmitex ora decommentato.

## Differenze con `project_root/experiments/run_keff_vs_delta.py`

### Il nostro script (PROBLEMA):
```python
# Crea pori UNIFORMI in tutto il volume
sphIncl_pores.fromHisto(seed, sac_de_billes.TypeAlgo.BOOL, 0., [[inclR, inclPhi_input]], [incl_phase])

# Overlay semplice
structure = merope.Structure_3D(multiInclusions_pores, multiInclusions_grains, {2:0, 3:0})
```

**Risultato**: Delta NON influenza la morfologia → K_eff varia poco e in modo errato.

### Gli script in test_por/ (CORRETTO):
- Separano **inter** (tra grani) e **intra** (dentro grani)
- Il boundary layer `delta` influenza la **distribuzione spaziale** dei pori inter
- Delta piccolo → boundaries sottili → pori inter formano rete interconnessa → bassa K_eff
- Delta grande → boundaries spessi → pori inter più isolati → alta K_eff

## Prossimi Passi

Per sistemare `run_keff_vs_delta.py`:

1. **Separare inter e intra pores** come in `2_rad_mixed_gen.py`
2. **Usare la formula di fit** `porosity = alpha*inclPhi + beta*delta + gamma` per bilanciare i contributi
3. **Doppio overlay** per combinare correttamente le tre componenti (grains, inter, intra)
4. Probabilmente serve **calibrare** i parametri alpha, beta, gamma per il nostro caso specifico

## Note Tecniche

- **VoxelRule.Average**: tutti gli script usano Average + Voigt homogenization
- **Resolution**: n3D=300 per risultati accurati (vs nostro n3D=100)
- **TypeAlgo**: BOOL per i pori, RSA per i grani Laguerre
- **Seed management**: cambiano seed tra iterazioni per evitare fallimenti RSA

---

**Conclusione**: Il segreto sta nel **separare inter e intra porosity** e fare in modo che delta influenzi principalmente la distribuzione dei pori inter-granulari, creando la transizione da morfologia interconnessa a distribuita.
