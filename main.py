from improved_rank_list import ImprovedRankList

def main():
    rank_list = ImprovedRankList()
    rank_list.parse_config('config/config.ini')
    rank_list.initialize()
    rank_list.run()
if __name__ == '__main__':
    main()
