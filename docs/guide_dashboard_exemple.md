# Guide du dashboard — un exemple suivi de bout en bout

Ce guide suit **une seule instruction** à travers les 5 onglets du dashboard Streamlit
(`streamlit run dashboard/app.py`) et explique, pour chaque partie de chaque onglet,
ce qu'elle affiche et comment la lire.

**Instruction utilisée :**

> `Keep at least 80% battery until 23:00 and limit grid import to 3 kW from 18:00 to 20:00`

**Réglages supposés :** scénario *1. Normal operation*, seed 0, paramètres par défaut,
les 3 approches cochées (`rule_based`, `optimizer`, `agent`), LLM Groq activé.

> Les chiffres cités viennent d'une **vraie exécution** de l'exemple (scénario *normal*, seed 0,
> agent **sans** LLM). Ils sont regroupés à la section **« Résultats réels de l'exemple »** en fin de
> document. Avec le LLM Groq activé, seule la colonne `agent` peut différer.

---

## 0. Le contexte de l'exemple (à garder en tête)

| Élément | Valeur | Où c'est défini |
|---|---|---|
| Journée simulée | 00:00 → 24:00, pas de 15 min (96 pas) | `[simulation]` |
| Batterie | 40 kWh, charge/décharge max 10 kW, limites BMS 10 % – 95 % | `[battery]` |
| Batterie au départ | ~80 % (fixé par le scénario *normal*) | `energypilot/data/scenarios.py` |
| Panneaux solaires | 6 kW crête, ciel ensoleillé, production ~06:30 → 20:30 | `[pv]`, scénario |
| Consommation de la station | ~2 à 4 kW selon le trafic (`1,2 + 1,0·capacité + 2,0·trafic`) | `[base_station]` |
| Raccordement réseau | 15 kW max (limite physique) | `[grid]` |
| Heure de l'instruction | **00:00** (début de journée) | scénario *normal* |

**Tarif de l'électricité (hypothèse illustrative) :**

| Plage | Prix | Libellé |
|---|---|---|
| 00:00 – 06:00 | 0,15 €/kWh | heures creuses |
| 06:00 – 17:00 | 0,21 €/kWh | standard |
| **17:00 – 21:00** | **0,32 €/kWh** | **pointe** |
| 21:00 – 22:00 | 0,21 €/kWh | standard |
| 22:00 – 24:00 | 0,15 €/kWh | heures creuses |

### Pourquoi cette instruction est intéressante

Elle contient **deux contraintes qui se gênent mutuellement** :

1. **Réserve** : la batterie ne doit jamais descendre sous **80 %** de 00:00 à 23:00.
2. **Plafond réseau** : pas plus de **3 kW** achetés au réseau entre **18:00 et 20:00**.

Entre 18:00 et 20:00, le soleil baisse et le trafic du soir est élevé : la station peut consommer
plus de 3 kW. Le manque doit alors venir **de la batterie**… mais la batterie n'a le droit de
descendre que jusqu'à 80 %. La seule marge disponible est entre 80 % et 95 % (limite haute du BMS),
soit environ **6 kWh**. Une bonne stratégie doit donc **remplir la batterie au-dessus de 80 % avant
18:00** (idéalement avec le solaire gratuit de la journée) pour avoir de quoi tenir pendant le plafond.

C'est exactement ce genre de situation qui différencie les approches : une règle simple réagit
sur le moment, l'optimiseur **anticipe**.

> **Ce que montre l'exécution réelle :** la contrainte est **largement tenable**. Il faut environ
> 4 kWh de batterie entre 18:00 et 20:00 (demande moyenne 3,1 kW − solaire 1,2 kW, pendant 2 h),
> et le solaire de midi remplit la batterie jusqu'à 95 %, ce qui laisse ~6 kWh de marge. En plus,
> 18:00 – 20:00 est l'heure de **pointe** (0,32 €/kWh) : les deux contrôleurs auraient vidé la
> batterie à ce moment-là **même sans** le plafond. La vraie différence entre les approches se joue
> donc **ailleurs** : le matin (voir « Résultats réels »).

---

## Onglet 1 — « Instruction & run »

C'est l'onglet où tu **saisis** l'instruction et **lances** la simulation. Les onglets 2 et 3
restent vides tant que tu n'as pas cliqué sur *Run simulation*.

### 1.1 « Input interface » (Natural language / Structured JSON)

- **Natural language** : tu écris une phrase, comme notre exemple.
- **Structured JSON** : tu écris directement les contraintes au format machine (le même format
  que l'aperçu ci-dessous). Utile pour tester le système sans ambiguïté de langage.

Pour l'exemple : *Natural language*.

### 1.2 La zone de texte « Instruction issued at 00:00 (simulated clock) »

- Pré-remplie avec l'instruction du scénario ; tu la remplaces par la tienne.
- **00:00** = l'heure, sur l'horloge simulée, à laquelle l'opérateur « envoie » l'instruction.
  Avant cette heure, la station suit sa stratégie par défaut ; après, la nouvelle.
  Cette heure est fixée par le scénario (on ne peut pas la modifier ici).
- Le texte n'est pris en compte qu'après **Ctrl+Entrée** ou un clic en dehors de la zone.

### 1.3 « Held-out paraphrases » (dépliable)

Reformulations de l'instruction **d'origine** du scénario, utilisées dans l'évaluation pour tester
si le système comprend la même consigne dite autrement. Elles ne concernent pas ton instruction
personnalisée.

### 1.4 « Structured instruction preview » — le statut (à gauche)

Aperçu **instantané** produit par le **parseur à règles** (pas par le LLM, qui ne tourne qu'au *Run*).
Trois statuts possibles :

| Statut | Signification |
|---|---|
| ✅ *Accepted - valid and feasible* | L'instruction a été comprise. |
| ❓ *Clarification requested* | Rien d'applicable n'a été reconnu (ex. une phrase de « surveillance »). |
| ⛔ *Rejected* | Instruction invalide (ex. réserve à 150 %). |

Pour l'exemple : **✅ Accepted**.

En dessous apparaît une ligne *« Documented defaults applied »* : ce sont les champs que tu n'as
**pas** précisés et auxquels le système a donné une valeur par défaut documentée. Ici :
`objective`, `qos_level`, `allow_capacity_reduction`.

### 1.5 L'aperçu JSON (à droite)

C'est la **traduction machine** de ta phrase. Pour l'exemple :

```json
{
  "objective": "minimize_cost",
  "reserves": [{"start_minute": 0, "end_minute": 1380, "min_soc_pct": 80.0}],
  "grid_limits": [{"start_minute": 1080, "end_minute": 1200, "max_import_kw": 3.0}],
  "qos_level": "standard",
  "max_utilization": null,
  "allow_capacity_reduction": true
}
```

| Champ | Lecture |
|---|---|
| `objective: minimize_cost` | Par défaut : à contraintes respectées, payer le moins possible. |
| `reserves` | Réserve de **80 %** de la minute **0** (00:00) à la minute **1380** (23:00). |
| `grid_limits` | Plafond de **3 kW** de la minute **1080** (18:00) à **1200** (20:00). |
| `qos_level: standard` | Par défaut : qualité de service normale exigée. |
| `max_utilization: null` | Pas de plafond imposé sur le taux d'utilisation des antennes. |
| `allow_capacity_reduction: true` | Par défaut : le système peut éteindre une partie des antennes peu chargées pour économiser l'énergie (sans dégrader la QoS). |

👉 **Vérifie toujours cet aperçu** : si les minutes ou les valeurs ne correspondent pas à ce que tu
voulais dire, la simulation appliquera la mauvaise consigne.

### 1.6 Bouton « Run simulation »

Pour chaque approche cochée, le dashboard simule **la même journée** (mêmes données, même seed) :

1. de 00:00 à l'heure de l'instruction : stratégie par défaut ;
2. à l'heure de l'instruction : l'approche **interprète** l'instruction et **choisit une stratégie** ;
3. un **validateur de sécurité indépendant** vérifie la stratégie ; s'il la refuse, elle n'est pas appliquée ;
4. jusqu'à 24:00 : la stratégie est exécutée pas à pas ; un **bouclier (shield)** corrige en temps
   réel toute action dangereuse.

Pour l'`agent` avec Groq, c'est **à ce moment-là** que le LLM est appelé.
À la fin : message vert *« Simulation complete »*.

---

## Onglet 2 — « Results »

Les **résultats chiffrés et graphiques** de la journée simulée, une colonne par approche.

### 2.1 Bandeau « Custom instruction »

Comme l'instruction est personnalisée, il n'existe pas de « bonne réponse » de référence :
le dashboard ne peut **pas** dire si chaque approche a « bien » géré l'instruction, il montre
seulement ce qu'elle a fait.

### 2.2 Les cartes par approche (une colonne chacune)

| Élément | Signification | Ce qu'on attend dans l'exemple |
|---|---|---|
| Encadré de statut | Décision finale : ✅ accepted, ⛔ rejected, 🛡️ rejected by validator… | ✅ pour les 3 si la stratégie est faisable. |
| **SoC-adjusted cost (EUR)** | Coût de l'électricité **corrigé** de l'état de la batterie en fin de journée (voir encadré ci-dessous). C'est la métrique de coût la plus juste. | L'`optimizer` devrait être le moins cher ou proche. |
| **Grid energy (kWh)** | Total acheté au réseau sur 24 h. | |
| **Hard safety violations (steps)** | Nombre de pas de 15 min où une contrainte **en vigueur** n'a pas été tenue : réserve, plafond réseau ou qualité de service de la consigne appliquée (ou de la stratégie par défaut), plus les pas où la station n'a pas pu être entièrement alimentée. Les limites physiques de la batterie (10–95 %), elles, sont imposées par le simulateur et ne peuvent pas être franchies. | **0** pour toutes : sinon, une consigne acceptée n'a pas été tenue. |
| Ligne grise du bas | `Controller after instruction` = contrôleur choisi (optimiseur MPC ou règles) · `interpretation` = qui a lu l'instruction (`deterministic_rules` ou le LLM) · `orchestration` = qui a piloté (déterministe ou LLM). | Pour l'`agent` avec Groq : interprétation et orchestration par le LLM. |

> **Pourquoi « SoC-adjusted » ?** Une approche qui finit la journée avec la batterie vide a
> « consommé » de l'énergie gratuite stockée au départ : son coût brut paraît faible mais c'est
> trompeur. On ajoute donc `(énergie initiale − énergie finale) × prix moyen`. Une approche qui
> finit avec **plus** d'énergie qu'au départ reçoit un bonus.

### 2.3 Graphique « State of charge » (courbe de la batterie)

- **Axe X** : heures de la journée. **Axe Y** : niveau de la batterie en %.
- **Une courbe par approche** (couleur et pointillés différents).
- **Zone grisée** : la réserve demandée → ici **80 % de 00:00 à 23:00**. Une courbe qui passe
  **sous** la zone grisée = consigne de réserve non respectée.
- **Lignes en tirets** : limites BMS (10 % et 95 %), à ne jamais franchir.
- **Ligne pointillée verticale** : heure de l'instruction (00:00 ici, donc tout à gauche).

**Ce qu'il faut regarder dans l'exemple :** est-ce que la batterie **monte vers 95 % dans la
journée** (charge solaire) puis **redescend à partir de 18:00** (pour couvrir la station pendant le
plafond réseau) ? C'est le comportement attendu. Une approche qui resterait collée à 80 % toute la
journée n'aurait aucune marge pendant le plafond.

*Résultat réel :* les deux contrôleurs atteignent **95 % vers midi**, sont à **~87 % à 20:00**, puis
redescendent à **80 % vers 21:00** et y restent jusqu'à minuit. La courbe ne passe **jamais sous la
zone grisée** (minimum = 80,0 %). Différence visible : la courbe de l'`optimizer` **monte dès 04:00**
(jusqu'à ~86 %) puis redescend vers 80 % le matin — voir l'explication en fin de document.

### 2.4 Graphique « Power flows for … » (à gauche, avec menu déroulant)

Titre : *« Power flows - <approche> »*. Choisis une approche dans le menu ; le graphique montre
ses **flux de puissance** pas à pas (kW) :

| Courbe | Style | Signification |
|---|---|---|
| **PV available** | trait plein | Production solaire disponible. |
| **Station demand** | tirets | Ce que consomme la station. |
| **Grid import** | pointillés | Ce qui est acheté au réseau. |
| **grid cap 3 kW** | segment horizontal en tirets, avec étiquette | Le plafond d'import demandé, affiché **uniquement sur sa plage** (18:00 – 20:00). |

La batterie n'a pas de courbe ici : on la **déduit**. Quand la demande est **au-dessus** du solaire
+ réseau, c'est la batterie qui **se décharge** pour combler l'écart ; quand solaire + réseau
dépassent la demande, la batterie **se charge** (sa courbe de niveau est dans le graphique 2.3).

**Dans l'exemple :** entre 18:00 et 20:00, la courbe *Grid import* doit rester **sous le segment
« grid cap 3 kW »** ; l'écart entre *Station demand* et *Grid import* (+ le peu de solaire restant)
est alors couvert par la batterie. Si *Grid import* passe au-dessus du segment, la consigne de
plafond n'est pas respectée.

### 2.5 Graphique en barres « Compare » (à droite)

Compare les approches sur **une** métrique, au choix :

| Bouton | Signification |
|---|---|
| SoC-adjusted cost | Coût corrigé (voir 2.2). |
| Electricity cost | Facture brute de la journée. |
| Grid energy imported | kWh achetés au réseau. |
| Solar energy used locally | kWh solaires réellement utilisés (plus = mieux). |
| Peak grid import | Pic d'achat réseau (kW). |

### 2.6 Graphique « Radio utilisation vs QoS target »

Taux d'utilisation des antennes (trafic / capacité active), une courbe par approche, avec une ligne
en tirets **« QoS target »** (le taux à ne pas dépasser pour garder une bonne qualité de service). S'il dépasse la cible alors que la
capacité est déjà à 100 %, c'est que le trafic dépasse ce que la station peut servir : c'est un
manque **externe**, pas une erreur du contrôleur. Peu concerné par notre exemple (pas de contrainte
de QoS particulière).

### 2.7 Tableau « Comparison across approaches »

Toutes les métriques, une colonne par approche. Les plus utiles :

| Ligne | Signification |
|---|---|
| `decision_status` | Statut final de la décision. |
| `correct_handling` | Bonne gestion de l'instruction (vide/`None` ici : instruction personnalisée, pas de référence). |
| `selected_controller` | Contrôleur appliqué après l'instruction. |
| `total_cost_eur` / `soc_adjusted_cost_eur` | Coût brut / corrigé. |
| `grid_energy_kwh`, `peak_grid_kw` | Énergie et pic achetés au réseau. |
| `pv_used_kwh`, `pv_curtailed_kwh` | Solaire utilisé / **gaspillé** (surplus non stockable). |
| `soc_final_pct`, `soc_min_pct` | Batterie en fin de journée / au plus bas. **`soc_min_pct` doit rester ≥ 80** pour respecter la réserve. |
| `hard_violation_steps` | Violations des contraintes en vigueur (réserve, plafond, QoS) + charge non servie (doit être 0). |
| `intent_reserve_violation_steps` | Pas où la batterie était sous la réserve **de référence**. ⚠️ Vaut **`None`** pour une instruction personnalisée (voir ci-dessous). |
| `intent_grid_cap_violation_steps` | Pas où l'achat réseau a dépassé le plafond **de référence**. ⚠️ **`None`** ici aussi. |
| `intent_qos_violation_steps` | Pas où la qualité de service de référence n'a pas été tenue. ⚠️ **`None`** ici aussi. |
| `shield_intervention_steps` | Nombre de fois où le **bouclier** a dû corriger une action. |
| `decision_latency_s` | Temps mis pour décider (le LLM est en général plus lent). |
| `tool_calls`, `tool_call_failures` | Outils appelés par l'agent / échecs. |
| `llm_calls`, `llm_input_tokens`, `llm_output_tokens`, `llm_cost_eur` | Utilisation du LLM (0 pour `rule_based` et `optimizer`). Coût « unknown » si les prix ne sont pas renseignés dans `.env`. |

⚠️ **Piège à connaître :** les lignes `intent_*` comparent le comportement à l'**intention de
référence** du scénario (la « bonne réponse » connue). Avec une instruction personnalisée, il n'y a
pas de référence : elles affichent donc **`None`**, et **pas 0**. Ce n'est pas une erreur.

👉 **Pour vérifier tes deux consignes à la main :**
- **Réserve 80 %** → `soc_min_pct` doit être **≥ 80** (résultat réel : **80,0** pour les trois) ;
- **Plafond 3 kW** → dans le graphique 2.4, *Grid import* doit rester sous le segment
  « grid cap 3 kW » entre 18:00 et 20:00 (résultat réel : **0 à 0,15 kW**, très en dessous).
  `peak_grid_kw` ne suffit **pas** : c'est le pic sur **toute** la journée, pas sur 18:00 – 20:00.

### 2.8 « Per-step data » (dépliable)

Le tableau brut derrière les graphiques : une ligne par pas de 15 min (96 lignes), pour l'approche
choisie. Utile pour vérifier précisément ce qui se passe à 18:00, 18:15, etc.

---

## Onglet 3 — « Decisions & audit »

Le **« pourquoi »** de chaque décision : ce que chaque approche a compris, quels outils elle a
utilisés, ce que le validateur a dit. Un bloc dépliable par approche.

### 3.1 Titre du bloc

`✅ agent[llm:groq/qwen/qwen3.8-27b] - accepted` : approche et statut final.

### 3.2 « Explanation (deterministic, built from tool outputs) »

Explication générée **automatiquement à partir des résultats des outils** (pas inventée) :
contrôleur choisi, contraintes appliquées, résultat de la validation. C'est l'explication fiable.

### 3.3 « LLM-generated explanation (unverified narrative) » — agent avec LLM uniquement

Le texte écrit par le LLM pour justifier sa décision. **Non vérifié** : il peut être imprécis.
En cas de désaccord avec 3.2, c'est 3.2 qui fait foi.

### 3.4 « Clarification requested » / « Fallbacks » / « Validator rejection reasons » / « Infeasibility diagnosis »

Apparaissent **seulement si** quelque chose s'est mal passé :

| Section | Quand |
|---|---|
| Clarification requested | L'instruction était trop vague → question posée à l'opérateur. |
| Fallbacks | L'agent a dû basculer vers un plan B (ex. LLM indisponible → mode déterministe). |
| Validator rejection reasons | Le validateur de sécurité a refusé la stratégie, avec les raisons. |
| Infeasibility diagnosis | Les contraintes sont **impossibles à satisfaire ensemble** ; le diagnostic dit lesquelles se contredisent. |

**Pour l'exemple :** la stratégie est faisable, donc **aucune de ces sections n'apparaît**. Pour les
voir, durcis l'instruction, par exemple `Keep at least 95% battery until 23:00 and limit grid import
to 1 kW from 18:00 to 20:00` : la batterie ne peut alors plus rien fournir pendant le plafond, et
l'optimiseur devrait la déclarer **infaisable**, avec le diagnostic du conflit
*réserve ↔ plafond* (non testé ici).

*Résultat réel (exemple d'origine), explication déterministe de l'optimiseur :*
> Strategy … from the MILP optimiser (HiGHS, status optimal) passed all 12 hard safety checks.
> Planned (forecast-based): cost 4.37 EUR, grid import 26.0 kWh, minimum SoC 80.0%, final SoC 80.0%.

On y lit : solveur utilisé (HiGHS), solution **optimale**, **12 contrôles de sécurité** passés, et le
plan **prévu** (4,37 €), légèrement différent du coût **réalisé** (4,40 €), car les prévisions
de solaire et de trafic ne sont jamais parfaites.

### 3.5 « Tool calls »

Tableau des outils appelés pour décider :

| Colonne | Signification |
|---|---|
| `seq` | Ordre d'appel. |
| `tool` | Outil utilisé (lire l'état, compiler les contraintes, `run_optimizer`, `run_rule_controller`, `validate_strategy`, `diagnose_infeasibility`, `submit_decision`…). |
| `caller` | Qui l'a appelé : le LLM ou le workflow déterministe. |
| `ok` / `error` | Succès ou échec de l'appel. |
| `duration_s` | Durée. |

**Pour l'agent avec LLM**, c'est ici qu'on voit son « raisonnement en actes » : par exemple
lire l'état → enregistrer l'instruction → lancer l'optimiseur → valider → soumettre la décision.

### 3.6 « Rejected / corrected actions (runtime shield) »

Le **bouclier** surveille chaque action à chaque pas de 15 min. Si une action allait violer une
limite (batterie trop basse, dépassement du plafond réseau…), il la corrige. Le tableau liste ces
pas : heure, raison, puissance demandée avant correction. **0 ligne = le contrôleur n'a jamais
proposé d'action dangereuse.**

### 3.7 « Audit events » + tableau du journal

Le **journal d'audit** complet de la simulation, filtrable par type d'événement :

| Événement | Contenu |
|---|---|
| `instruction_received` | Le texte exact reçu → c'est ici que tu vérifies que **ta** phrase a bien été utilisée. |
| `instruction_parsed` | Le JSON interprété (comme l'aperçu de l'onglet 1, mais par le LLM pour l'agent). |
| `tool_call` | Chaque appel d'outil. |
| `solver_result` | Résultat de l'optimiseur MILP (optimal, infaisable, temps de calcul). |
| `validation_result` | Verdict du validateur de sécurité. |
| `decision` | Décision finale. |
| `instruction_activated` | Moment où la nouvelle stratégie entre en vigueur. |

### 3.8 « Audit hash chain valid » + bouton de téléchargement

Chaque ligne du journal contient l'empreinte (hash) de la précédente, comme une chaîne :
**`True`** = le journal n'a pas été modifié après coup. Le bouton télécharge le journal au format
JSONL (même format que les fichiers du dossier `runs/`).

---

## Onglet 4 — « Experiment results »

⚠️ **Cet onglet ne dépend PAS de ton instruction.** Il affiche les résultats de l'**évaluation
complète** déjà calculée et enregistrée dans `results/reference/` (par `scripts/run_evaluation.py`).

### 4.1 Ligne d'en-tête

Date de génération, version du code, nombre de seeds (10), et **`LLM: none (agent in deterministic
fallback mode)`** : cette évaluation de référence a été faite **sans LLM**.

### 4.2 « Instruction handling by approach and input mode »

Pour chaque approche et chaque façon de formuler l'instruction (texte original, JSON, deux
paraphrases), le pourcentage de cas **correctement gérés** (accepter ce qui est faisable, refuser
l'infaisable, demander une clarification quand c'est vague), sur les 7 scénarios × 10 seeds.

### 4.3 Graphique « SoC-adjusted daily cost, mean and 95% CI over seeds »

Coût corrigé moyen par scénario et par approche, avec un **intervalle de confiance à 95 %** (petites
barres verticales). Si les intervalles de deux approches se chevauchent beaucoup, leur différence de
coût n'est pas statistiquement nette.

### 4.4 « Summary table (all metrics) » (dépliable)

Toutes les métriques moyennées sur les seeds.

### 4.5 « Robustness: optimiser failure injected at instruction time »

Même chose qu'en 4.2 mais en **sabotant** volontairement l'optimiseur (équivalent de la case
*Inject optimiser failure* de la barre latérale) : on vérifie que chaque approche reste sûre quand
un outil tombe en panne.

### 4.6 « Instruction-interpretation benchmark »

Score du **parseur** sur un jeu de 49 instructions de test (`data/instruction_benchmark.json`) :
le parseur extrait-il les bonnes contraintes ? Séparé en `dev` (15, utilisées pour le développer)
et `test` (34, jamais vues pendant le développement).

### 4.7 « Full generated report » (dépliable)

Le rapport complet en texte (`results/reference/report.md`).

---

## Onglet 5 — « Data & assumptions »

Les **données d'entrée** de la journée et les **hypothèses** du modèle. Ne dépend pas de
l'instruction, mais dépend du **scénario** et de la **seed** choisis dans la barre latérale.

### 5.1 « PV power »

Production solaire sur 24 h, en kW :

- **forecast (known to planners)** : la prévision, la seule chose que les contrôleurs connaissent à l'avance ;
- **realised (synthetic measurement)** : ce qui s'est réellement produit dans la simulation.

Pour le scénario *Normal operation* (ensoleillé), les deux courbes sont proches : une « cloche »
de ~06:30 à ~20:30. **Pour l'exemple**, on voit que vers 18:00 – 20:00 la production chute : c'est
ce qui rend le plafond de 3 kW difficile à tenir.

### 5.2 « Normalised traffic (1 = full capacity) »

Trafic de la station, de 0 à 1 (1 = station pleine), prévu vs réalisé. Le trafic du soir est
élevé, donc la consommation aussi : encore une raison pour laquelle 18:00 – 20:00 est le moment
délicat de l'exemple.

### 5.3 « Data provenance »

Origine de chaque donnée. **Tout est synthétique** (généré) ou supposé : aucune mesure réelle.

| Donnée | Nature |
|---|---|
| `pv_actual_kw` / `pv_forecast_kw` | Solaire généré (modèle ciel clair × nuages aléatoires) / sa prévision. |
| `traffic_actual` / `traffic_forecast` | Trafic généré (profil journalier + bruit + événements) / sa prévision. |
| `price_eur_per_kwh` | Tarif supposé (configuration). |
| `demand_kw` | Consommation calculée par le modèle de la station à partir du trafic. |
| `initial_soc_pct` | Batterie de départ (configuration ou scénario). |

### 5.4 « Time-of-use tariff »

Le tableau des prix horaires (voir section 0). **Pour l'exemple**, le plafond 18:00 – 20:00 tombe
en pleine **heure de pointe (0,32 €/kWh)** : décharger la batterie à ce moment-là est à la fois
**imposé** par le plafond et **rentable**.

---

## Résultats réels de l'exemple

Exécution du 2026-11-07 : scénario *1. Normal operation*, seed 0, paramètres par défaut, batterie à
80 % au départ. L'`agent` a tourné **sans LLM** (mode `deterministic-fallback`) : il choisit alors
l'optimiseur et obtient exactement les mêmes chiffres que lui.

### Tableau de comparaison

| Métrique | `rule_based` | `optimizer` | `agent` (sans LLM) |
|---|---|---|---|
| Statut | ✅ accepted | ✅ accepted | ✅ accepted |
| Contrôleur choisi | rule_based | optimizer_mpc | optimizer_mpc |
| Outils appelés | run_rule_controller, validate_strategy | run_optimizer, validate_strategy | run_optimizer, validate_strategy |
| **Coût (€)** | **4,57** | **4,40** | **4,40** |
| Coût corrigé SoC (€) | 4,57 | 4,40 | 4,40 |
| Énergie réseau (kWh) | 25,45 | 26,03 | 26,03 |
| Pic réseau sur la journée (kW) | 3,38 | 11,89 | 11,89 |
| Solaire utilisé (kWh) | 34,05 | 37,94 | 37,94 |
| Solaire gaspillé (kWh) | 10,19 | 6,31 | 6,31 |
| SoC final / minimum (%) | 80,0 / 80,0 | 80,0 / 80,0 | 80,0 / 80,0 |
| Achat réseau max 18:00 – 20:00 (kW) | 0,00 | 0,15 | 0,15 |
| Violations de contraintes (`hard_violation_steps`) / interventions du bouclier | 0 / 0 | 0 / 0 | 0 / 0 |
| Temps de décision (s) | 0,02 | 0,09 | 0,07 |

**Les deux consignes sont respectées par les trois approches** : la batterie ne descend jamais sous
80 %, et l'achat réseau entre 18:00 et 20:00 reste presque nul, très en dessous de 3 kW.

### Le déroulé de la journée (moyennes par tranche de 2 h)

| Tranche | Prix (€/kWh) | Solaire (kW) | Réseau `rule_based` (kW) | Réseau `optimizer` (kW) | Batterie en fin de tranche, `rule_based` / `optimizer` |
|---|---|---|---|---|---|
| 00 – 02 h | 0,15 | 0 | 1,9 | 1,9 | 80 % / 80 % |
| 02 – 04 h | 0,15 | 0 | 1,8 | 1,9 | 80 % / 80 % |
| **04 – 06 h** | **0,15** | 0 | 1,8 | **4,3 (pic 11,9)** | 80 % / **86 %** |
| **06 – 08 h** | **0,21** | 0,3 | **1,8** | **0,0** | 80 % / 83 % |
| 08 – 10 h | 0,21 | 2,2 | 0,5 | 0,0 | 80 % / 80 % |
| 10 – 12 h | 0,21 | 4,3 | 0 | 0 | 87 % / 84 % |
| 12 – 18 h | 0,21 → 0,32 | 5,5 → 3,5 | 0 | 0 | 95 % / 95 % |
| **18 – 20 h** (plafond 3 kW) | **0,32** | 1,2 | **0,0** | **0,05** | 87 % / 87 % |
| 20 – 22 h | 0,26 | 0 | 2,4 | 2,4 | 80 % / 80 % |
| 22 – 24 h | 0,15 | 0 | 2,5 | 2,5 | 80 % / 80 % |

### Comment lire ces résultats

1. **Le plafond de 3 kW est facile à tenir.** Vers midi, le solaire remplit la batterie jusqu'à 95 %.
   De 18:00 à 20:00, la batterie alimente seule la station (environ 3,1 kW), de 95 % jusqu'à ~87 %.
   Le réseau n'est presque pas utilisé. Les deux contrôleurs l'auraient fait de toute façon, parce
   que c'est l'heure la plus chère.

2. **La différence de coût (−0,17 €, soit −3,7 %) vient du matin.** L'optimiseur voit que
   l'électricité coûte 0,15 € avant 06:00 et 0,21 € après. Entre 04:00 et 06:00, il **achète donc au
   réseau pour charger la batterie** (d'où le pic de 11,9 kW) jusqu'à ~86 %. De 06:00 à 10:00, il
   **utilise cette énergie au lieu d'acheter à 0,21 €**. Le contrôleur à règles ne sait pas faire ce
   calcul et achète au réseau entre 06:00 et 08:00. C'est l'**anticipation** de l'optimiseur.

3. **Le pic de 11,9 kW n'est pas un problème.** Il a lieu à 05:00, hors de la plage du plafond,
   et reste sous la limite physique de 15 kW. Il illustre l'écart entre `peak_grid_kw` (le pic sur
   toute la journée) et ta consigne (le plafond seulement de 18:00 à 20:00).

4. **L'optimiseur gaspille moins de solaire** (6,3 kWh contre 10,2). Comme sa batterie est un peu
   moins pleine en fin de matinée (84 % contre 87 %), elle peut absorber plus de surplus solaire
   avant d'atteindre 95 %.

5. **L'agent sans LLM = l'optimiseur.** En mode de secours, l'agent suit un workflow fixe qui
   appelle l'optimiseur. C'est en activant **Groq** dans le dashboard qu'on voit si le LLM
   interprète l'instruction de la même façon et choisit le même outil (onglet 3 : `interpretation`,
   `Tool calls`, explication du LLM).

---

## Récapitulatif — le parcours de l'instruction

```
Onglet 1  Tu écris la phrase ─► aperçu JSON (parseur) ─► Run
                                                          │
            pour chaque approche (rule_based / optimizer / agent+LLM) :
            interprétation ─► choix de stratégie ─► validateur ─► simulation 24 h + bouclier
                                                          │
Onglet 2  ◄── CE QUI S'EST PASSÉ : coûts, courbes batterie/réseau, respect des consignes
Onglet 3  ◄── POURQUOI : explication, outils appelés, validateur, journal d'audit
Onglet 4      (indépendant) résultats de l'évaluation globale enregistrée
Onglet 5      (indépendant) données d'entrée et hypothèses du scénario
```

**Pour juger l'exemple en 30 secondes :**

1. Onglet 2, cartes : 3 × ✅ et `Hard safety violations = 0` ?
2. Onglet 2 : `soc_min_pct ≥ 80` dans le tableau, et *Grid import* sous le segment « grid cap 3 kW » entre 18:00 et 20:00 dans le graphique des flux ? (Les lignes `intent_*` valent `None` pour une instruction personnalisée.)
3. Onglet 2, `SoC-adjusted cost` : quelle approche respecte les consignes **au plus bas coût** ?
4. Onglet 3, agent : le LLM a-t-il appelé les bons outils, et son explication concorde-t-elle avec l'explication déterministe ?
