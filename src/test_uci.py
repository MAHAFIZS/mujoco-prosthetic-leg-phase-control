from src.uci_loader import load_uci


data = load_uci()

print("Train shape:", data["X_train"].shape)
print("Test shape:", data["X_test"].shape)

print("Unique labels:", sorted(set(data["y_train"])))
print("Activities:")
for k, v in data["activity_labels"].items():
    print(k, ":", v)
