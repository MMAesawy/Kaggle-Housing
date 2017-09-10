print("Grid searching GB model...")
gb_gs = GridSearchCV(cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1,
                      estimator=GradientBoostingRegressor(),
                      param_grid=[{'learning_rate': [x / 100 for x in range(5, 15, 1)],
                                   'max_depth': [x for x in range(2, 5)],
                                   'max_features': [x / 10 for x in range(7, 11)]}])
gb_gs.fit(X_train, y_train)
print(f'Best score is {(-gb_gs.best_score_)**.5}')
print(gb_gs.best_params_)
print()

print("Grid searching bagging model...")
bag_gs = GridSearchCV(cv=10, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1,
                      estimator=BaggingRegressor(base_estimator = LinearRegression(),
                                                 n_estimators = 25),
                      param_grid=[{'max_features': [x / 100 for x in range(50, 101, 1)]}])
bag_gs.fit(X_train, y_train)
print(f'Best score is {(-bag_gs.best_score_)**.5}')
print(bag_gs.best_params_)
print()

print("Grid searching lasso model...")
lasso_gs = GridSearchCV(cv=10, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1,
                      estimator=Lasso(max_iter = 10**5),
                      param_grid=[{'alpha': [x for x in range(1, 1000, 10)]}])
lasso_gs.fit(X_train, y_train)
print(f'Best score is {(-lasso_gs.best_score_)**.5}')
print(lasso_gs.best_params_)
print()

print("Grid searching ridge model...")
ridge_gs = GridSearchCV(cv=10, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1,
                      estimator=Ridge(max_iter = 10**5),
                      param_grid=[{'alpha': [x for x in range(0, 1000, 10)]}])
ridge_gs.fit(X_train, y_train)
print(f'Best score is {(-ridge_gs.best_score_)**.5}')
print(ridge_gs.best_params_)
print()

print("Grid searching RF model... ")
rf_gs = GridSearchCV(cv=10, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1,
                      estimator=RandomForestRegressor(n_estimators = 100,
                                                       max_features='sqrt',
                                                       min_samples_split=5,
                                                       random_state=random_seed),
                      param_grid=[{'max_depth': [x for x in range(2, 50)]}])
rf_gs.fit(X_train, y_train)
print(f'Best score is {(-rf_gs.best_score_)**.5}')
print(rf_gs.best_params_)

print("Grid searching KNN model... ")
knn_gs = GridSearchCV(cv=10, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1,
                      estimator=KNeighborsRegressor(),
                      param_grid=[{'n_neighbors': [x for x in range(3, 80)],
                                   'weights': ['uniform', 'distance']}])
knn_gs.fit(X_train, y_train)
print(f'Best score is {(-knn_gs.best_score_)**.5}')
print(knn_gs.best_params_)