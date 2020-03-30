#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'

from datetime import datetime


GET_IMAGE_DATA = """
select
product.daddtime,
product.validend,
product.sproductid,
product.spicurl,
category1.icategoryid as firstcatid,
category2.icategoryid as secondcatid,
category3.icategoryid as thirdcatid, 
category1.scategory as firstcat,
category2.scategory as secondcat,
category3.scategory as thirdcat,
brand.ibrandid as ibrandid,
brand.sbrand as sbrand,
brand.sbranden as sbranden
from Ymt_Products product
join ymt_productcategory category3 on product.ithirdcategoryid = category3.icategoryid
join ymt_productcategory category2 on category2.icategoryid = category3.iMasterCategory
join ymt_productcategory category1 on category1.icategoryid = category2.iMasterCategory
join ymt_productbrand brand on product.ibrandid = brand.ibrandid
where product.validEnd > '{0}'
""".format(datetime.now().strftime("%Y/%-m/%-d %-H:%-M:%-S"))

GET_IMAGE_HASH = """SELECT pid, imgmd5 from c2csearchEngine.dbo.productprofile a  with (nolock)
join (SELECT sproductid FROM integratedproduct.dbo.Ymt_Products with (nolock) where validend > dateadd(hour, -2, getdate())) b on a.pid=b.sproductid where imgmd5 is not null
"""
