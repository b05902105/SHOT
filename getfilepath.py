from OfficeHome import *

if __name__ == '__main__':
    names = ['Art', 'Clipart', 'Product', 'RealWorld']
    data_forder = '/tmp2/yc980802/da/data/OfficeHome'

    for n in names:
        with open(n+'.txt', 'w') as f:
            f.write(gen_path(data_forder, n))
        
