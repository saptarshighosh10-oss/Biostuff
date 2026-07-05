"""
Demonstration benchmark — textbook extreme cases.

A curated held-out set the model NEVER saw during training:
  - Positives: canonical aggregation-prone / amyloidogenic proteins from the
    literature (Alzheimer's, Parkinson's, type-2 diabetes, prion disease, etc.)
  - Negatives: classic well-behaved, highly soluble proteins used as folding
    and expression standards.

Purpose: show that the model's learned notion of "aggregation risk" —
trained only on antibody developability data — still ranks unambiguous
aggregators above unambiguous soluble proteins. This is a cross-domain
generalization test, not a claim of clinical accuracy.

Sequences are canonical forms from UniProt / primary literature. Where the
aggregating species is a fragment (e.g. Abeta42, PrP106-126), the fragment
is used because that is the unit that actually aggregates.
"""

# ── Confirmed aggregators (label = confirmed_failure) ────────────────────────
AGGREGATORS = [
    ("Abeta42",        "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
                       "Alzheimer amyloid-beta 1-42; the canonical amyloid"),
    ("Abeta40",        "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVV",
                       "Alzheimer amyloid-beta 1-40"),
    ("alpha_synuclein","MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
                       "Parkinson alpha-synuclein; Lewy body aggregator"),
    ("IAPP_amylin",    "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY",
                       "Islet amyloid polypeptide; type-2 diabetes"),
    ("PrP_106_126",    "KTNMKHMAGAAAAGAVVGGLG",
                       "Prion protein neurotoxic fragment 106-126"),
    ("insulin_Bchain", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
                       "Insulin B-chain; aggregates/fibrillates under stress"),
    ("calcitonin",     "CGNLSTCMLGTYTQDFNKFHTFPQTAIGVGAP",
                       "Human calcitonin; forms amyloid fibrils"),
    ("polyQ40",        "Q" * 40,
                       "Polyglutamine tract (Q40); Huntington-type aggregation"),
    ("amyloidogenic_TTR105","YTIAALLSPYSYSTTAVVTN",
                       "Transthyretin 105-115 amyloidogenic segment"),
    ("serum_amyloid_A","GFFSFIGEAFQGAGDMWRAYTDMKEAGWKDGDKYFHARGNYDAA",
                       "Serum amyloid A N-terminal amyloidogenic region"),
]

# ── Well-behaved soluble proteins (label = working) ──────────────────────────
SOLUBLE = [
    ("ubiquitin",      "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                       "Ubiquitin; textbook stable, highly soluble fold"),
    ("protein_G_B1",   "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                       "Protein G B1 domain; folding-standard, very stable"),
    ("trastuzumab_VH", "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
                       "Trastuzumab (Herceptin) VH; approved, well-formulated"),
    ("trastuzumab_VL", "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK",
                       "Trastuzumab VL; approved therapeutic"),
    ("cytochrome_c",   "GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
                       "Horse cytochrome c; stable soluble heme protein"),
    ("lysozyme_C",     "KVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV",
                       "Human lysozyme C; secreted soluble enzyme"),
    ("myoglobin",      "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG",
                       "Sperm-whale myoglobin; classic soluble globin"),
    ("thioredoxin",    "SDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
                       "E. coli thioredoxin; high-solubility expression tag"),
    ("GB1_gfp_stable", "MASKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
                       "GFP; folds cleanly, highly soluble reporter"),
    ("barstar",        "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS",
                       "Barstar; small soluble ribonuclease inhibitor"),
]


def load_benchmark() -> list[dict]:
    """
    Return the full benchmark as candidate dicts ready for scoring.
    Each dict has variant_sequence, label, name, note, source.
    """
    entries = []
    for name, seq, note in AGGREGATORS:
        entries.append({
            "variant_sequence": seq.upper(),
            "label": "confirmed_failure",
            "name": name,
            "note": note,
            "source": "benchmark",
            "mutations": [],
        })
    for name, seq, note in SOLUBLE:
        entries.append({
            "variant_sequence": seq.upper(),
            "label": "working",
            "name": name,
            "note": note,
            "source": "benchmark",
            "mutations": [],
        })
    return entries
