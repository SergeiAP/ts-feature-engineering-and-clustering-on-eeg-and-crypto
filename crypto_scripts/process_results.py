import pandas as pd


def get_non_btc_tickets(df: pd.DataFrame,
                        target_clusters: dict[str, int],
                        df_names: list[str]) -> list[set]:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        target_clusters (dict[str, int]): _description_
        df_names (list[str]): _description_

    Returns:
        list[set]: _description_
    """
    non_btc_tickets = []
    for df_name in df_names:
        print(f"Different from BTC currencies in cluster {df_name}:")
        non_btc_ticket = set(
            df[df_name]['cluster'][
                df[df_name]['cluster'] != target_clusters[df_name]].index)
        non_btc_tickets.append(non_btc_ticket)
        print(f"Not like BTC tickets {df_name}:", 
              len(non_btc_ticket), '\n', non_btc_ticket)
    return non_btc_tickets
