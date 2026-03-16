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
