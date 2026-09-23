--> Aula 1 - Boas Práticas de Código Limpo em ML

## Resumo

A aula discute por que, em projetos de Machine Learning, "código funcionar" não é o mesmo que "código bem construído". O código é o elemento que sustenta todo o pipeline (carregamento de dados, pré-processamento, treinamento, avaliação) e define diretamente como o modelo aprende. Um modelo pode ter métricas aparentemente boas mesmo tendo sido treinado de forma incorreta (ex: vazamento de dados, transformações inconsistentes). Por isso, aplicar princípios de Clean Code não é uma questão estética, mas uma forma de garantir confiabilidade, auditabilidade e escalabilidade das soluções de ML.

## Conceitos-chave

- **Clean Code**: escrever código claro, simples e bem estruturado, tornando explícitas as transformações, a lógica do modelo e as decisões do pipeline.
- **Código que funciona x código que escala**: código funcional apenas resolve o problema imediato; código que escala é auditável, reprodutível e permite incorporar novos dados/modelos sem reescrever tudo.
- **Legibilidade**: escrever para humanos, não só para a máquina — nomes claros, organização lógica e separação de responsabilidades reduzem o esforço cognitivo e o risco de erros.
- **Nomes significativos**: variáveis, funções e módulos devem expressar sua intenção/finalidade, não detalhes de implementação. Reduz ambiguidade e facilita manutenção.
- **Princípio KISS** (*Keep It Simple, Stupid*): resolver o problema com o menor nível de complexidade necessário; evitar abstrações prematuras e complexidade desnecessária.
- **Princípio DRY** (*Don't Repeat Yourself*): cada regra/transformação deve existir em um único lugar no código, evitando divergências entre etapas do pipeline.
- **Code Review**: revisão do código por outras pessoas antes da integração; essencial para identificar erros silenciosos (que não quebram a execução, mas invalidam os resultados) e padronizar o código.

## Anotações

- Em ML, erros conceituais (vazamento de dados, avaliação incorreta) não geram falhas técnicas visíveis — o sistema "roda normalmente", mas produz resultados inválidos. Isso torna Clean Code ainda mais crítico do que em software tradicional.
- Comentários devem explicar **decisões conceituais**, não compensar código mal escrito. Se o código é claro, a própria estrutura já comunica o necessário.
- Dividir o pipeline em etapas isoladas (dados → pré-processamento → treino → avaliação) facilita testes, depuração e modificações localizadas.
- Auditabilidade é cada vez mais exigida por regulamentações e questões éticas em sistemas automatizados — Clean Code vira parte da governança do modelo, não só boa prática de estilo.
- Tendência de mercado: Clean Code deixou de ser recomendação individual e passou a ser requisito organizacional, puxado por MLOps/DevOps (integração contínua, versionamento, testes automatizados).

### Case: Knight Capital (2012)
- Empresa de trading de alta frequência colocou uma atualização em produção; um módulo antigo que deveria estar desativado foi executado por engano.
- Resultado: prejuízo de US$ 440 milhões em menos de 45 minutos, quase levando a empresa à falência.
- Causa raiz: código legado mal estruturado, falta de isolamento de funcionalidades, ausência de testes e falhas no processo de code review — não um erro matemático complexo.
- Usado como exemplo de que código "estatisticamente correto" ainda pode causar risco operacional inaceitável se não for claro, revisado e bem estruturado.

## Questões e dúvidas

- 
- 

## Recursos

- MARTIN, R. C. *Clean Code: a handbook of agile software craftsmanship*. Prentice Hall, 2008.
- MARTIN, R. C. *Clean Architecture*. Prentice Hall, 2017.
- FOWLER, M. *Refactoring: improving the design of existing code*. 2. ed. Addison-Wesley, 2018.
- SCULLEY, D. et al. *Hidden technical debt in machine learning systems*. NeurIPS, 2015.
- GOOGLE. *Engineering practices documentation: code review*. https://google.github.io/eng-practices/review/
- GOOGLE. *Machine Learning Engineering Best Practices*. https://developers.google.com/machine-learning/guides/rules-of-ml
- GOOGLE CLOUD. *MLOps: continuous delivery and automation pipelines in machine learning*.
- MICROSOFT. *MLOps principles*.
- SCIKIT-LEARN. *Pipelines and composite estimators*.
- WALL STREET MILLENNIAL. *A Hedge Fund Accidentally Used The Wrong Code, Lost $400 Million* (YouTube, 2021).

## Próximos passos

- Revisar um projeto de ML pessoal aplicando KISS e DRY
- [ ] Praticar renomeação de variáveis/funções em um pipeline existente
- [ ] Configurar um processo simples de code review (mesmo que solo, revisar código no dia seguinte)
- [ ] Ler o case Knight Capital com mais profundidade como estudo de risco operacional
