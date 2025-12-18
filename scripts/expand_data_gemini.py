import pandas as pd

# 1. Deine existierenden Dateien laden
# (Falls du Google Colab nutzt, lade die Dateien erst links im Menü hoch!)
try:
    df_clinical = pd.read_csv("./data/synthetic/german_safety_v2_clinical.csv")
    df_answers = pd.read_csv("./data/synthetic/german_safety_v2_answers.csv")
except FileNotFoundError:
    print("FEHLER: Bitte lade erst die CSV-Dateien hoch!")

# 2. Die fehlenden "Harten Fälle" (Manual Injection)
# Diese decken Selbstverletzung, Methoden und Abschied ab (die fehlten bisher).
new_unsafe_texts = [
    "Ich schneide mich heute tief.",
    "Das Blut muss fließen, damit der Druck weggeht.",
    "Habe die Klingen schon parat im Bad.",
    "Ich nehme das Messer und mache dem ein Ende.",
    "Ich ritze mir die Arme auf.",
    "Ich gehe heute Nacht an die Gleise.",
    "Wenn der nächste Zug kommt, springe ich.",
    "Ich werfe mich vor die Bahn.",
    "Ich habe ein Seil gekauft.",
    "Ich erhänge mich auf dem Dachboden.",
    "Macht's gut, ich kann nicht mehr. Das ist mein letzter Eintrag.",
    "Ich habe Abschiedsbriefe geschrieben, morgen bin ich weg.",
    "Keiner wird mich vermissen, wenn ich tot bin.",
    "Ich fahre mit dem Auto gegen den Baum.",
    "Ich springe von der Brücke."
]

repair_data = [
    # --- FIX FÜR PIZZA (Aktive Alltags-Handlungen -> SAFE) ---
    {"text": "Ich esse heute Pizza.", "label": "SAFE"},
    {"text": "Ich koche mir extra viele Nudeln.", "label": "SAFE"},
    {"text": "Ich gehe gleich alles einkaufen.", "label": "SAFE"},
    {"text": "Ich habe ein Glas Wasser getrunken.", "label": "SAFE"},
    {"text": "Ich putze meine Wohnung.", "label": "SAFE"},
    {"text": "Ich werde ein Buch lesen.", "label": "SAFE"},
    {"text": "Ich fahre nie mit dem Bus zur Arbeit.", "label": "SAFE"},

    # --- FIX FÜR "NICHT MEHR LEBEN" (Negation als Gefahr -> UNSAFE) ---
    {"text": "Ich will nicht mehr leben.", "label": "UNSAFE"},
    {"text": "Ich möchte nicht mehr da sein.", "label": "UNSAFE"},
    {"text": "Es wäre besser, wenn ich nicht mehr aufwache.", "label": "UNSAFE"},
    {"text": "Ich halte es nicht mehr aus, ich will weg.", "label": "UNSAFE"},
    {"text": "Für mich gibt es kein Morgen mehr.", "label": "UNSAFE"}
]

df_repair = pd.DataFrame(repair_data)

# Erstellen des neuen Dataframes für die harten Fälle
df_new_unsafe = pd.DataFrame({
    'text': new_unsafe_texts,
    'label': ['UNSAFE'] * len(new_unsafe_texts)
})

# 3. Downsampling der App-Antworten (SAFE)
# Wir nehmen nur 50 Stück, damit das Verhältnis bei ca. 1:2 bleibt.
df_answers_sample = df_answers.sample(n=75, random_state=42)
df_answers_sample['label'] = 'SAFE_APP_CONTEXT' 
# Wir brauchen nur die Text-Spalte (und Label) - Spalte umbenennen von 'answer' zu 'text'
df_answers_sample = df_answers_sample[['answer', 'label']].rename(columns={'answer': 'text'})

# 4. Alles zusammenfügen
# Wir reduzieren df_clinical auch auf die relevanten Spalten, damit es passt
df_clinical_clean = df_clinical[['text', 'label']]

# df_repair hinzufügen (enthält sowohl SAFE als auch UNSAFE Repair-Daten)
df_final = pd.concat([df_clinical_clean, df_new_unsafe, df_repair, df_answers_sample], ignore_index=True)

# 5. Mapping auf 0 und 1 (Für SetFit Training)
# 1 = ALARM (Unsafe)
# 0 = RUHE (Alles andere: Safe, Mundane, Clinical Symptoms)
def map_to_binary(label):
    if label == 'UNSAFE':
        return 1
    return 0

df_final['label'] = df_final['label'].apply(map_to_binary)

# 6. Einmal gut durchmischen (Shuffle)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# 7. Speichern
output_filename = "./data/synthetic/final_setfit_train.csv"
df_final.to_csv(output_filename, index=False)

# Statistik ausgeben
print(f"✅ Datei '{output_filename}' erfolgreich erstellt!")
print("-" * 30)
counts = df_final['label'].value_counts()
print(f"SAFE (0):   {counts.get(0, 0)}")
print(f"UNSAFE (1): {counts.get(1, 0)}")
ratio = counts.get(0, 0) / counts.get(1, 0)
print(f"Verhältnis: 1 : {ratio:.2f} (Ideal für SetFit)")