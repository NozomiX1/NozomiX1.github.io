# 极简个人主页

一个「陈列项目 + GitHub 仓库」的个人主页。排版提炼自 `report/index.html`：衬线字体、黑白灰墨色、三线表、细边框，靠留白撑起阅读感。

## 结构

```
site/
├── index.html           主页：关于 + 项目 + 仓库
├── css/
│   └── style.css        唯一一份设计系统，全站共用
├── fonts/
│   ├── noto-serif-sc-vf-subset.woff2  自托管思源宋体网页子集
│   ├── OFL.txt                         字体许可证
│   └── README.md                       字体来源与更新方法
├── reports/             技术报告页面
├── scripts/
│   └── build_font_subset.py            字体子集生成脚本
└── README.md
```

纯 HTML，零依赖、零构建。双击 `index.html` 即可本地预览。

## 主页两个区块的区别

- **项目**：作品 / 成果展示，不一定要是代码仓库（比如研究报告、系统、Demo），标题是正常的中文名。
- **仓库**：专门陈列 GitHub repositories，标题是等宽字体的 `用户名/仓库名`。

## 部署到 GitHub Pages

1. 在 GitHub 新建仓库，名字必须是 `你的用户名.github.io`。

2. 把仓库根目录整理成下面这样，再 push（`index.html` 必须在根目录）：

   ```
   你的用户名.github.io/
   ├── index.html          ← 来自 site/index.html
   ├── css/style.css       ← 来自 site/css/style.css
   └── report/             ← 可选：项目里链到的技术报告
   ```

   ```bash
   git init
   git add .
   git commit -m "init"
   git remote add origin https://github.com/你的用户名/你的用户名.github.io.git
   git push -u origin main
   ```

3. 仓库 **Settings → Pages**，Source 选 `main` → Save，等 1～2 分钟。

   访问 `https://你的用户名.github.io`。

## 添加一个项目

在 `index.html` 的 `.projectlist` 里复制一块：

```html
<article class="project">
  <div class="p-head">
    <h3 class="p-name"><a href="#">项目名</a></h3>
    <span class="p-links"><a href="#">GitHub</a><a href="#">Demo</a></span>
  </div>
  <p class="p-desc">一句话说清楚它是什么、解决了什么问题。</p>
  <div class="p-tags">语言 · 框架 · 关键词</div>
</article>
```

## 添加一个仓库

在 `index.html` 的 `.repolist` 里复制一块：

```html
<article class="repo">
  <div class="r-head">
    <h3 class="r-name"><a href="https://github.com/yourname/repo">yourname/repo</a></h3>
    <span class="r-links"><a href="https://github.com/yourname/repo">GitHub</a></span>
  </div>
  <p class="r-desc">一句话描述。</p>
  <div class="r-meta">Python · ⭐ 12 · 关键词</div>
</article>
```

## 自定义

- **名字 / 简介 / 关键词 / GitHub 链接**：改 `index.html` 顶部 `.titleblock` 和「关　于」区块（已留注释）。
- **颜色 / 字体**：改 `css/style.css` 顶部的 `:root` 变量。

## 更新网页字体

主页和技术报告共用 `fonts/noto-serif-sc-vf-subset.woff2`。字体已经预留
GB2312 一级常用汉字；如果新增较多生僻字或符号，重新生成一次：

```powershell
python -m pip install "fonttools[woff]==4.59.2"
python scripts/build_font_subset.py
```

字体来源、许可证和覆盖范围见 `fonts/README.md`。

## 一个建议

主页不需要一开始就填满。先放一个真实项目（比如那个检索 Agent），以后每做完一个往里加一块，慢慢就充实了。
