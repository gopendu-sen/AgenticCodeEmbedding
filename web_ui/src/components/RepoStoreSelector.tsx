interface RepoStoreSelectorProps {
  stores: string[];
  selectedStores: string[];
  manualStoreInput: string;
  onSelectStores: (stores: string[]) => void;
  onManualInputChange: (value: string) => void;
  onRefresh: () => Promise<void>;
}


export function RepoStoreSelector({
  stores,
  selectedStores,
  manualStoreInput,
  onSelectStores,
  onManualInputChange,
  onRefresh
}: RepoStoreSelectorProps) {
  const toggle = (store: string) => {
    const next = selectedStores.includes(store)
      ? selectedStores.filter((s) => s !== store)
      : [...selectedStores, store];
    onSelectStores(next);
  };

  return (
    <section className="sidebar-card">
      <div className="sidebar-card-header">
        <h2>Repo Stores</h2>
        <button type="button" onClick={() => void onRefresh()}>
          Refresh
        </button>
      </div>
      {stores.length ? (
        <ul className="store-list">
          {stores.map((store) => (
            <li key={store}>
              <label>
                <input
                  type="checkbox"
                  checked={selectedStores.includes(store)}
                  onChange={() => toggle(store)}
                />
                <span>{store}</span>
              </label>
            </li>
          ))}
        </ul>
      ) : (
        <div className="manual-store">
          <p>No stores discovered yet. Add comma-separated store names manually.</p>
          <input
            value={manualStoreInput}
            onChange={(event) => onManualInputChange(event.target.value)}
            placeholder="repo_a, repo_b"
          />
        </div>
      )}
    </section>
  );
}
