# -*- coding: utf-8 -*-
#PYGAME_FREETYPE = True  
import os
import pygame
from pygame import ftfont



def draw(c, idx, directory, lang):
    pygame.init()
    #font = pygame.font.Font("./fonts/NotoSansCJKsc-Black.otf", 36)
    if lang == 'zh_traditional':
        font = ftfont.Font("./fonts/NotoSansCJKtc-Regular.otf", 36)
    elif lang == 'zh_simplified':
        font = ftfont.Font("./fonts/NotoSansCJKsc-Regular.otf", 36)
    elif lang == 'ja':
        font = ftfont.Font("./fonts/NotoSansCJKjp-Regular.otf", 36)
    elif lang == 'ko':
        font = ftfont.Font("./fonts/NotoSansCJKkr-Regular.otf", 36)
    rtext = font.render(c, True, (0, 0, 0), (255, 255, 255))
    rtext = pygame.transform.scale(rtext, (36, 36))
    pygame.image.save(rtext, directory + str(idx) + ".jpg")
